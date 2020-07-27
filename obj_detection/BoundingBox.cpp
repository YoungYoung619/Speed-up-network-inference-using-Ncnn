#include "BoundingBox.h"

#include <cstdarg>
#include <vector>


ObjDet::BoundingBox::BoundingBox()
{
}

ObjDet::BoundingBox::BoundingBox(BoxFormat format, float v1, float v2, float v3, float v4)
{	

	if (format == X1X2Y1Y2) {
		x1 = v1;
		y1 = v2;
		x2 = v3;
		y2 = v4;

		x = (x2 + x1) / 2.0;
		y = (y2 + y1) / 2.0;
		h = y2 - y1;
		w = x2 - x1;
	}
	else if (format == CXCYHW){
		x = v1;
		y = v2;
		w = v3;
		h = v4;
		/*printf("%f %f %f %f", x, y, h, w);
		printf("\n");*/

		x1 = x - w / 2.f;
		x2 = x + w / 2.f;
		y1 = y - h / 2.f;
		y2 = y + h / 2.f;
	}
	else {
		throw "NotImplementedException";
	}
}


ObjDet::BoundingBox::~BoundingBox()
{

}

std::vector<float> ObjDet::BoundingBox::x1x2y1y2()
{	
	throw "NotImplementedException";
	return std::vector<float>();
}

std::vector<float> ObjDet::BoundingBox::cxcyhw()
{	
	throw "NotImplementedException";
	return std::vector<float>();
}
