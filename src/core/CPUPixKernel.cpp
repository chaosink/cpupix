#include "CPUPix.hpp"

#include <iostream>
#include <algorithm>
#include <unordered_set>

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
	int x_l[], int x_r[],
	Fragment fragment_l[], Fragment fragment_r[]
) {
	int x = x0, dx = x1 - x0;
	int y = y0, dy = y1 - y0;
	assert(dy >= 0);
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
		if(x_l[y - y0] > x) {
			x_l[y - y0] = x;
			fragment_l[y - y0] = f;
		}
		if(x_r[y - y0] < x) {
			x_r[y - y0] = x;
			fragment_r[y - y0] = f;
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
	if(x_l[y1 - y0] > x1) {
		x_l[y1 - y0] = x1;
		fragment_l[y1 - y0] = f1;
	}
	if(x_r[y1 - y0] < x1) {
		x_r[y1 - y0] = x1;
		fragment_r[y1 - y0] = f1;
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

	int *x_l = new int[p[2].y - p[0].y + 1];
	int *x_r = new int[p[2].y - p[0].y + 1];
	Fragment *fragment_l = new Fragment[p[2].y - p[0].y + 1];
	Fragment *fragment_r = new Fragment[p[2].y - p[0].y + 1];
	std::fill(x_r, x_r + p[2].y - p[0].y + 1, -1);
	std::fill(x_l, x_l + p[2].y - p[0].y + 1, w);

	AssemSegment(
		p[0].x, p[0].y,
		p[1].x, p[1].y,
		Fragment{v[0].position.z, v[0].position.w, vo[0]},
		Fragment{v[1].position.z, v[1].position.w, vo[1]},
		x_l, x_r,
		fragment_l, fragment_r);
	AssemSegment(
		p[0].x, p[0].y,
		p[2].x, p[2].y,
		Fragment{v[0].position.z, v[0].position.w, vo[0]},
		Fragment{v[2].position.z, v[2].position.w, vo[2]},
		x_l, x_r,
		fragment_l, fragment_r);
	AssemSegment(
		p[1].x, p[1].y,
		p[2].x, p[2].y,
		Fragment{v[1].position.z, v[1].position.w, vo[1]},
		Fragment{v[2].position.z, v[2].position.w, vo[2]},
		x_l + p[1].y - p[0].y, x_r + p[1].y - p[0].y,
		fragment_l + p[1].y - p[0].y, fragment_r + p[1].y - p[0].y);

	for(int y = p[0].y; y <= p[2].y; ++y) {
		if(y < 0 || y >= h) continue;
		int i = y - p[0].y;
		if(x_r[i] < x_l[i]) continue;
		if(x_r[i] < 0 || x_l[i] >= w) continue; // maybe needless if do 2D clipping
		int l = x_l[i];
		int r = glm::min(x_r[i], w - 1);
		Fragment fragment = fragment_l[i];
		Fragment fragment_delta = (fragment_r[i] - fragment_l[i]) / (x_r[i] - x_l[i]);
		if(x_l[i] < 0) {  // maybe needless if do 2D clipping
			l = 0;
			fragment += fragment_delta * -x_l[i];
		}
		scanline[y].segment.push_back(Segment{
			l,
			r - l + 1,
			fragment,
			fragment_delta
		});
	}

	delete[] x_l;
	delete[] x_r;
	delete[] fragment_l;
	delete[] fragment_r;
}

void DrawPixel(int x, int y, Fragment &fragment, float *depth_buf, unsigned char* frame_buf) {
	if(fragment.z > 1 || fragment.z < -1) return; // need 3D clipping
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

int IntersectSegment(Segment &s0, Segment &s1) {
	return
		((s0.fragment_delta.z * s0.x - s1.fragment_delta.z * s1.x)
			- (s0.fragment.z - s1.fragment.z))
		/ (s0.fragment_delta.z - s1.fragment_delta.z);
}

// scanline with segment if depth_test is true
void DrawSegmentWithDepthTest(Scanline *scanline, float *depth_buf, unsigned char* frame_buf) {
	#pragma omp parallel for
	for(int y = 0; y < h; ++y) {
		std::vector<Segment> &seg = scanline[y].segment;
		if(seg.size() == 0) continue;

		std::vector<ScanNode> node;
		node.reserve(seg.size() * 2);
		for(size_t i = 0; i < seg.size(); ++i) {
			node.push_back(ScanNode{
				true,
				seg[i].x,
				// &seg[i]
				static_cast<int>(i)
			});
			node.push_back(ScanNode{
				false,
				seg[i].x + seg[i].length - 1,
				// &seg[i]
				static_cast<int>(i)
			});
		}
		std::sort(node.begin(), node.end(), [](const ScanNode &n0, const ScanNode &n1){
			return n0.x < n1.x || (n0.x == n1.x && n0.in && !n1.in);
		});

		std::unordered_set<int> segment_in;
		int segment;
		Fragment fragment;
		for(size_t i = 0; i < node.size() - 1; ++i) {
			if(segment_in.empty()) {
				assert(node[i].in);
				segment_in.insert(node[i].segment);
				segment = node[i].segment;
				fragment = seg[segment].fragment;
				for(int x = node[i].x; x < node[i + 1].x;
						++x, fragment += seg[segment].fragment_delta)
					DrawPixel(x, y, fragment, depth_buf, frame_buf);
			} else {
				if(node[i].in)
					segment_in.insert(node[i].segment);
				else
					segment_in.erase(node[i].segment);
				if(segment_in.empty()) continue;

				// scanline with segment for segment_in.size() == 1 or 2
				if(segment_in.size() == 1) {
					segment = *segment_in.begin();
					fragment = seg[segment].f(node[i].x);
					for(int x = node[i].x; x < node[i + 1].x;
							++x, fragment += seg[segment].fragment_delta)
						DrawPixel(x, y, fragment, depth_buf, frame_buf);
					continue;
				} else if(segment_in.size() == 2) {
					int s0 = *segment_in.begin(), s1 = *(++segment_in.begin());
					if(seg[s0].z(node[i].x) > seg[s1].z(node[i].x)) // need z clipping
						std::swap(s0, s1);
					if(seg[s0].z(node[i + 1].x) > seg[s1].z(node[i + 1].x)) { // need z clipping
						int ix = IntersectSegment(seg[s0], seg[s1]);
						segment = s0;
						fragment = seg[segment].f(node[i].x);
						for(int x = node[i].x; x < ix;
								++x, fragment += seg[segment].fragment_delta)
							DrawPixel(x, y, fragment, depth_buf, frame_buf);
						segment = s1;
						fragment = seg[segment].f(ix);
						for(int x = ix; x < node[i + 1].x;
								++x, fragment += seg[segment].fragment_delta)
							DrawPixel(x, y, fragment, depth_buf, frame_buf);
						continue;
					} else {
						segment = s0;
						fragment = seg[segment].f(node[i].x);
						for(int x = node[i].x; x < node[i + 1].x;
								++x, fragment += seg[segment].fragment_delta)
							DrawPixel(x, y, fragment, depth_buf, frame_buf);
						continue;
					}
					continue;
				}

				// ordinary scanline for segment_in.size() > 2
				for(auto s: segment_in) {
					int x = node[i].x;
					Fragment fragment = seg[s].f(x);
					for(; x < node[i + 1].x;
							++x, fragment += seg[s].fragment_delta)
						DrawPixel(x, y, fragment, depth_buf, frame_buf);
				}
			}
		}

		seg.clear();
	}
}

// ordinary scanline if depth_test is false
void DrawSegmentWithoutDepthTest(Scanline *scanline, float *depth_buf, unsigned char* frame_buf) {
	#pragma omp parallel for
	for(int y = 0; y < h; ++y) {
		std::vector<Segment> &seg = scanline[y].segment;
		for(size_t i = 0; i < seg.size(); ++i) {
			int x = seg[i].x;
			Fragment fragment = seg[i].fragment;
			for(int k = 0; k < seg[i].length - 1;
					++k, ++x, fragment += seg[i].fragment_delta)
				DrawPixel(x, y, fragment, depth_buf, frame_buf);
		}
		seg.clear();
	}
}

void DrawSegment(Scanline *scanline, float *depth_buf, unsigned char* frame_buf) {
	if(depth_test)
		DrawSegmentWithDepthTest(scanline, depth_buf, frame_buf);
	else
		DrawSegmentWithoutDepthTest(scanline, depth_buf, frame_buf);
}

void DrawCharater(int ch, int x0, int y0, unsigned char *frame_buf) {
	#pragma omp parallel for
	for(int y = 0; y < 16; ++y)
		for(int x = 0; x < 16; ++x) {
			int i_pixel = w * (y + y0) + x + x0;
			int offset = ch * 32;
			char c = bitmap[offset + (15 - y) * 2 + x / 8];
			if(!(c & bit[x % 8])) continue;
			frame_buf[i_pixel * 3 + 0] = 255 - clear_color[0];
			frame_buf[i_pixel * 3 + 1] = 255 - clear_color[1];
			frame_buf[i_pixel * 3 + 2] = 255 - clear_color[2];
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
