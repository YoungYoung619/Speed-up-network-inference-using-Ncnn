#pragma once
#include <vector>

namespace ObjDet {
	enum BoxFormat { CXCYHW, X1X2Y1Y2 };

	class BoundingBox
	{
	public:
		BoundingBox();
		BoundingBox(BoxFormat format, float v1, float v2, float v3, float v4);
		~BoundingBox();

		std::vector<float> x1x2y1y2();
		std::vector<float> cxcyhw();

		// vars
		float x1;
		float x2;
		float y1;
		float y2;

		float x;
		float y;
		float h;
		float w;
	};
}