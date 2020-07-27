#pragma once
#include "BoundingBox.h"


namespace ObjDet {
	enum ClassType { Vehilce, Person };

	class ObjDetItem
	{
	public:
		ObjDetItem();
		~ObjDetItem();

		//avrs
		ObjDet::BoundingBox box;
		ObjDet::ClassType classtype;
		float score;
		int idx;
	};
}


