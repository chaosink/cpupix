#include "CPUPix.hpp"

#include <iostream>
using namespace std;

namespace cpupix {

namespace kernel {

extern int w, h;
extern int n_triangle;
extern bool depth_test;
extern bool blend;
extern glm::u8vec4 clear_color;

unsigned char bit[8] = {
	0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
extern char bitmap[];

void VertexShader(VertexIn &in, VertexOut &out, Vertex &v);
void FragmentShader(FragmentIn &in, glm::vec4 &color);

void Clear(unsigned char *frame_buf, float *depth_buf) {
	#pragma omp parallel for
	for(int y = 0; y < h; ++y)
		for(int x = 0; x < w; ++x) {
			int i_pixel = y * w + x;
			frame_buf[i_pixel * 3 + 0] = clear_color.r;
			frame_buf[i_pixel * 3 + 1] = clear_color.g;
			frame_buf[i_pixel * 3 + 2] = clear_color.b;
			depth_buf[i_pixel] = 0;
		}
}

void NormalSpace(VertexIn *in, VertexOut *out, Vertex *v) {
	#pragma omp parallel for
	for(int x = 0; x < n_triangle * 3; ++x) {
		VertexShader(in[x], out[x], v[x]);

		float w_inv = 1.f / v[x].position.w; // divide w beforehand
		v[x].position.w = w_inv;
		out[x].position *= w_inv;
		out[x].normal   *= w_inv;
		out[x].uv       *= w_inv;
	}
}

void WindowSpace(Vertex *v) {
	#pragma omp parallel for
	for(int x = 0; x < n_triangle * 3; ++x) {
		glm::vec3 p = v[x].position * v[x].position.w;
		p.x = (p.x * 0.5f + 0.5f) * w;
		p.y = (p.y * 0.5f + 0.5f) * h;
		v[x].position.x = p.x;
		v[x].position.y = p.y;
		v[x].position.z = p.z;
	}
}

void AssemTriangle(Vertex *v, Triangle *triangle) {
	#pragma omp parallel for
	for(int x = 0; x < n_triangle; ++x) {
		glm::vec2
			p0(v[x * 3 + 0].position.x, v[x * 3 + 0].position.y),
			p1(v[x * 3 + 1].position.x, v[x * 3 + 1].position.y),
			p2(v[x * 3 + 2].position.x, v[x * 3 + 2].position.y);
		glm::vec2
			v_min = glm::min(glm::min(p0, p1), p2),
			v_max = glm::max(glm::max(p0, p1), p2);
		glm::ivec2
			iv_min = v_min + 0.5f,
			iv_max = v_max + 0.5f;

		triangle[x].empty = (iv_min.x >= w || iv_min.y >= h || iv_max.x < 0 || iv_max.y < 0
			|| (v[0].position.z > 1 && v[1].position.z > 1 && v[2].position.z > 1)
			|| (v[0].position.z <-1 && v[1].position.z <-1 && v[2].position.z <-1));
		triangle[x].winding = Winding((p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x) < 0);
	}
}

void AssemSegment(
	int x0, int y0,
	int x1, int y1,
	Fragment &&f0, Fragment &&f1,
	int border_x_l[], int border_x_r[],
	Fragment border_fragment_l[], Fragment border_fragment_r[]
) {
	int x = x0, dx = x1 - x0;
	int y = y0, dy = y1 - y0; // dy >= 0 is guaranteed
	if(dy == 0) return;

	Fragment f = f0, df = (f1 - f0) / dy;

	float xx = x;
	float dxx = dx * 1.f / dy;

	// int d = 0; // Bresenham
	// if(dx > dy) {
	// 	df = (f1 - f0) / float(x1 - x0);
	// 	f = f0;
	// } else {
	// 	df = (f1 - f0) / float(y1 - y0);
	// 	f = f0;
	// }

	while(y != y1) {
		if(border_x_l[y - y0] > x) {
			border_x_l[y - y0] = x;
			border_fragment_l[y - y0] = f;
		}
		if(border_x_r[y - y0] < x) {
			border_x_r[y - y0] = x;
			border_fragment_r[y - y0] = f;
		}
		y++;
		xx += dxx;
		x = xx;
		f += df;
		// if(dx > dy) { // Bresenham
		// 	d += 2 * dy;
		// 	if(d > dx) {
		// 		y += y1 > y0 ? 1 : -1;
		// 		d -= 2 * dx;
		// 	}
		// 	x += x1 > x0 ? 1 : -1;
		// 	f += x1 > x0 ? df : -df;
		// } else {
		// 	d += 2 * dx;
		// 	if(d > dy){
		// 		x += x1 > x0 ? 1 : -1;
		// 		d -= 2 * dy;
		// 	}
		// 	y += y1 > y0 ? 1 : -1;
		// 	f += y1 > y0 ? df : -df;
		// }
	}
	if(border_x_l[y1 - y0] > x1) {
		border_x_l[y1 - y0] = x1;
		border_fragment_l[y1 - y0] = f1;
	}
	if(border_x_r[y1 - y0] < x1) {
		border_x_r[y1 - y0] = x1;
		border_fragment_r[y1 - y0] = f1;
	}
}

void ScanTriangle(Vertex *v, VertexOut *vo, Scanline *scanline) {
	if(v[0].position.y > v[1].position.y) {
		std::swap(v[0], v[1]);
		std::swap(vo[0], vo[1]);
	}
	if(v[0].position.y > v[2].position.y) {
		std::swap(v[0], v[2]);
		std::swap(vo[0], vo[2]);
	}
	if(v[1].position.y > v[2].position.y) {
		std::swap(v[1], v[2]);
		std::swap(vo[1], vo[2]);
	}
	glm::ivec2 p[3] = {
		glm::ivec2(v[0].position.x, v[0].position.y),
		glm::ivec2(v[1].position.x, v[1].position.y),
		glm::ivec2(v[2].position.x, v[2].position.y)};
	if(p[0].y >= h || p[2].y < 0) return;

	int border_x_l[p[2].y - p[0].y + 1], border_x_r[p[2].y - p[0].y + 1];
	Fragment border_fragment_l[p[2].y - p[0].y + 1], border_fragment_r[p[2].y - p[0].y + 1];
	std::fill(border_x_r, border_x_r + p[2].y - p[0].y + 1, -1);
	std::fill(border_x_l, border_x_l + p[2].y - p[0].y + 1, w);

	AssemSegment(
		p[0].x, p[0].y,
		p[1].x, p[1].y,
		Fragment{v[0].position.z, v[0].position.w, vo[0]},
		Fragment{v[1].position.z, v[1].position.w, vo[1]},
		border_x_l, border_x_r,
		border_fragment_l, border_fragment_r);
	AssemSegment(
		p[0].x, p[0].y,
		p[2].x, p[2].y,
		Fragment{v[0].position.z, v[0].position.w, vo[0]},
		Fragment{v[2].position.z, v[2].position.w, vo[2]},
		border_x_l, border_x_r,
		border_fragment_l, border_fragment_r);
	AssemSegment(
		p[1].x, p[1].y,
		p[2].x, p[2].y,
		Fragment{v[1].position.z, v[1].position.w, vo[1]},
		Fragment{v[2].position.z, v[2].position.w, vo[2]},
		border_x_l + p[1].y - p[0].y, border_x_r + p[1].y - p[0].y,
		border_fragment_l + p[1].y - p[0].y, border_fragment_r + p[1].y - p[0].y);

	for(int y = p[0].y; y <= p[2].y; ++y) {
		if(y < 0 || y >= h) continue;
		int i = y - p[0].y;
		scanline[y].segment.push_back(Segment{
			border_x_l[i],
			border_x_r[i] - border_x_l[i],
			border_fragment_l[i],
			(border_fragment_r[i] - border_fragment_l[i]) / (border_x_r[i] - border_x_l[i] )
		});
	}
}

void DrawSegment(Scanline *scanline, float *depth_buf, unsigned char* frame_buf) {
	#pragma omp parallel for
	for(int y = 0; y < h; ++y) {
		for(size_t i = 0; i < scanline[y].segment.size(); ++i) {
			int x = scanline[y].segment[i].x - 1;
			Fragment fragment = scanline[y].segment[i].fragment - scanline[y].segment[i].fragment_delta;
			for(int k = 0; k < scanline[y].segment[i].length; ++k) {
				x++;
				fragment += scanline[y].segment[i].fragment_delta;
				if(x < 0 || x >= w) continue;
				if(fragment.z > 1 || fragment.z < -1) continue; // need 3D clipping
				int i_pixel = y * w + x;
				if(!depth_test || 1 - fragment.z > depth_buf[i_pixel]) {
					depth_buf[i_pixel] = 1 - fragment.z;
					glm::vec4 color;
					FragmentIn fi{
						glm::vec2(x, y),
						fragment.z,
						fragment.vo.position / fragment.w,
						fragment.vo.normal / fragment.w,
						fragment.vo.uv / fragment.w,
					};
					FragmentShader(fi, color);
					glm::ivec4 icolor;
					if(blend) {
						float alpha = color.a;
						glm::vec4 color_old = glm::vec4(
							frame_buf[i_pixel * 3 + 0],
							frame_buf[i_pixel * 3 + 1],
							frame_buf[i_pixel * 3 + 2],
							0.f
						);
						icolor = color * 255.f * alpha + color_old * (1.f - alpha);
					} else {
						icolor = color * 255.f;
					}
					icolor = glm::clamp(icolor, glm::ivec4(0), glm::ivec4(255));
					frame_buf[i_pixel * 3 + 0] = icolor.r;
					frame_buf[i_pixel * 3 + 1] = icolor.g;
					frame_buf[i_pixel * 3 + 2] = icolor.b;
				}
			}
		}
		scanline[y].segment.clear();
	}
}

void DrawCharater(int ch, int x0, int y0, bool aa, unsigned char *frame_buf) {
	#pragma omp parallel for
	for(int y = 0; y < 16; ++y)
		for(int x = 0; x < 16; ++x) {
			int i_pixel = w * (y + y0) + x + x0;
			int offset = ch * 32;
			char c = bitmap[offset + (15 - y) * 2 + x / 8];
			if(!(c & bit[x % 8])) continue;
			if(aa) {
				int p0 = (((y + y0) * 2 + 0) * w + (x + x0) * 2 + 0) * 3;
				int p1 = (((y + y0) * 2 + 0) * w + (x + x0) * 2 + 1) * 3;
				int p2 = (((y + y0) * 2 + 1) * w + (x + x0) * 2 + 0) * 3;
				int p3 = (((y + y0) * 2 + 1) * w + (x + x0) * 2 + 1) * 3;
				frame_buf[p0 + 0] = frame_buf[p1 + 0] = frame_buf[p2 + 0] = frame_buf[p3 + 0] = 255 - clear_color[0];
				frame_buf[p0 + 1] = frame_buf[p1 + 1] = frame_buf[p2 + 1] = frame_buf[p3 + 1] = 255 - clear_color[1];
				frame_buf[p0 + 2] = frame_buf[p1 + 2] = frame_buf[p2 + 2] = frame_buf[p3 + 2] = 255 - clear_color[2];
			} else {
				frame_buf[i_pixel * 3 + 0] = 255 - clear_color[0];
				frame_buf[i_pixel * 3 + 1] = 255 - clear_color[1];
				frame_buf[i_pixel * 3 + 2] = 255 - clear_color[2];
			}
		}
}

void DownSample(unsigned char *frame_buf, unsigned char *pbo_buf) {
	#pragma omp parallel for
	for(int y = 0; y < h / 2; ++y)
		for(int x = 0; x < w / 2; ++x) {
			int i_pixel = w / 2 * y + x;

			int p0 = ((y * 2 + 0) * w + x * 2 + 0) * 3;
			int p1 = ((y * 2 + 0) * w + x * 2 + 1) * 3;
			int p2 = ((y * 2 + 1) * w + x * 2 + 0) * 3;
			int p3 = ((y * 2 + 1) * w + x * 2 + 1) * 3;

			int r = (frame_buf[p0 + 0] + frame_buf[p1 + 0] + frame_buf[p2 + 0] + frame_buf[p3 + 0]) / 4;
			int g = (frame_buf[p0 + 1] + frame_buf[p1 + 1] + frame_buf[p2 + 1] + frame_buf[p3 + 1]) / 4;
			int b = (frame_buf[p0 + 2] + frame_buf[p1 + 2] + frame_buf[p2 + 2] + frame_buf[p3 + 2]) / 4;

			pbo_buf[i_pixel * 3 + 0] = glm::clamp(r, 0, 255);
			pbo_buf[i_pixel * 3 + 1] = glm::clamp(g, 0, 255);
			pbo_buf[i_pixel * 3 + 2] = glm::clamp(b, 0, 255);
		}
}

}

}
