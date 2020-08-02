// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "ncnn/net.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
#include<time.h>

#include "ncnn/blob.h"
#include "ncnn/layer.h"

#include "BoundingBox.h"
#include "ObjDetItem.h"



#define __max(a, b) (((a) > (b)) ? (a) : (b))
#define __min(a, b) (((a) < (b)) ? (a) : (b))

static const char* class_names[] = { "vehicle",
									"person",
									"traffic light",
									"traffic sign"
};

static const int n_class = 4;	// vehicle, person, traffic light, traffic sign
static const int n_anchor = 7;
static const int n_dim_per_bbox = n_class + 5; //(x1, y1, x2, y2, conf, cls_0, ...,cls_n)

static cv::Scalar lable_color[n_class] = { cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 0), cv::Scalar(204, 50, 153) };

static float anchors[7][2] = {
	{1.075f, 0.85f},
	{4.25f, 4.4625f},
	{0.3625f, 0.9f},
	{1.6875f, 1.3f},
	{0.675f, 1.625f},
	{2.3875f, 2.35f},
	{0.7f, 0.5625f}
};


template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
	return std::distance(first, std::max_element(first, last));
}

float sigmoid(float x)
{
	return (1 / (1 + exp(-x)));
}

std::vector<float> softmax(std::vector<float> logits) {
	float sum = 0.0;
	for (size_t i = 0; i < logits.size(); i++) {
		sum += exp(logits[i]);
	}

	for (size_t i = 0; i < logits.size(); i++) logits[i] = exp(logits[i])/sum;
	return logits;
}

float calc_iou(ObjDet::ObjDetItem rect1, ObjDet::ObjDetItem rect2)
{
	float xx1, yy1, xx2, yy2;

	xx1 = __max(rect1.box.x1, rect2.box.x1);
	yy1 = __max(rect1.box.y1, rect2.box.y1);
	xx2 = __min(rect1.box.x2, rect2.box.x2);
	yy2 = __min(rect1.box.y2, rect2.box.y2);

	float insection_width, insection_height;
	insection_width = __max(0.f, xx2 - xx1);
	insection_height = __max(0.f, yy2 - yy1);

	float insection_area, union_area, iou;
	insection_area = float(insection_width) * insection_height;
	union_area = __max(float(rect1.box.w*rect1.box.h + rect2.box.w*rect2.box.h - insection_area), (float) 1e-6);
	iou = insection_area / union_area;
	return iou;
}

bool cmpScore(ObjDet::ObjDetItem x, ObjDet::ObjDetItem y)
{
	if (x.score > y.score)
		return true;
	else
		return false;
}

std::vector<ObjDet::ObjDetItem> nms_boxes(std::vector<ObjDet::ObjDetItem> &bboxes, float confThreshold, float nmsThreshold)
{
	std::sort(bboxes.begin(), bboxes.end(), cmpScore);

	for (int i = 0; i < bboxes.size(); i++) {
		if (bboxes[i].score >= confThreshold){
			for (int j = i + 1; j < bboxes.size(); j++) {
				if (bboxes[j].score > confThreshold) {
					float iou = calc_iou(bboxes[i], bboxes[j]);
					if (iou > nmsThreshold){
						bboxes[j].score = 0.f;
					}
				}
				else {
					bboxes[j].score = 0.f;
				}
			}
		}
		else {
			bboxes[i].score = 0.f; //标记为抑制
		}
	}

	for (auto it = bboxes.begin(); it != bboxes.end();) {
		if ((*it).score == 0.f)
			bboxes.erase(it);//清除被抑制的候选框
		else
			it++;
	}

	return bboxes;
}


std::vector<ObjDet::ObjDetItem> nms_boxes_v2(std::vector<ObjDet::ObjDetItem>&bboxes, float confThreshold, float nmsThreshold)
{
	std::vector<ObjDet::ObjDetItem> results;
	std::sort(bboxes.begin(), bboxes.end(), cmpScore);
	while (bboxes.size() > 0)
	{
		results.push_back(bboxes[0]);
		int index = 1;
		while (index < bboxes.size()) {
			float iou_value = calc_iou(bboxes[0], bboxes[index]);
			if (bboxes[index].score < confThreshold or iou_value > nmsThreshold)
				bboxes.erase(bboxes.begin() + index);
			else
				index++;
		}
		bboxes.erase(bboxes.begin());
	}

	return results;
}

std::vector<ObjDet::BoundingBox> decode_bboxes(ncnn::Mat output) {

}

void draw_objects(const cv::Mat& bgr, const std::vector<ObjDet::ObjDetItem>& objects)
{	
	cv::Mat image = bgr.clone();

	for (size_t i = 0; i < objects.size(); i++)
	{
		const ObjDet::ObjDetItem& obj = objects[i];

		int x1 = (int)(obj.box.x1*1280.f);
		int y1 = (int)(obj.box.y1*720.f);
		int x2 = (int)(obj.box.x2*1280.f);
		int y2 = (int)(obj.box.y2*720.f);
		cv::rectangle(image, cvPoint(x1, y1), cvPoint(x2, y2), lable_color[obj.classtype], 4);
		printf("x1:%d  y1:%d  x2:%d  y2:%d conf:%f  class_idx:%d\n", x1, y1, x2, y2, obj.score, obj.classtype);
		char text[256];
		sprintf_s(text, "%s %.1f%%", class_names[obj.classtype], obj.score * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int x = (int) (obj.box.x1*1280.f);
		int y = (int) (obj.box.y1*720.f) - label_size.height - baseLine;
		if (y < 0)
			y = 0;
		if (x + label_size.width > image.cols)
			x = image.cols - label_size.width;

		cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
			cv::Scalar(255, 255, 255), -1);

		cv::putText(image, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}

	cv::imshow("detection", image);
	cv::waitKey(0);
}

std::vector<ObjDet::ObjDetItem> detect(const cv::Mat& bgr)
{	
	ncnn::Net mynet;

	mynet.opt.use_vulkan_compute = true;

	// the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
	/*mynet.load_param("E:\\\\software\\\\ncnn-master\\\\build-vs2017\\\\tools\\\\onnx\\\\bdd100k_det.param");
	mynet.load_model("E:\\\\software\\\\ncnn-master\\\\build-vs2017\\\\tools\\\\onnx\\\\bdd100k_det.bin");*/

	mynet.load_param("E:\\\\software\\\\ncnn-master\\\\build-vs2017\\\\tools\\\\quantize\\\\bdd100k_det_int8.param");
	mynet.load_model("E:\\\\software\\\\ncnn-master\\\\build-vs2017\\\\tools\\\\quantize\\\\bdd100k_det_int8.bin");

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_RGB, bgr.cols, bgr.rows, 320, 180);

	const float mean_vals[3] = { 104.f, 107.f, 123.f };
	in.substract_mean_normalize(mean_vals, 0);

	ncnn::Extractor ex = mynet.create_extractor();
	ex.set_num_threads(6);
	ex.input("input.1", in);

	ncnn::Mat out;
	ex.extract("180", out);
	printf("%d %d %d\n", out.w, out.h, out.c);
	
	std::vector<std::vector<float>> bboxes; // x1y1x2y2
	bboxes.resize(out.w*out.h * 7);

	for (int q = 0; q < out.c; q++)
	{
		const float* ptr = out.channel(q);
		for (int y = 0; y < out.h; y++)
		{
			for (int x = 0; x < out.w; x++)
			{
				int idx = y*out.w*n_anchor + n_anchor *x + (int) q / n_dim_per_bbox;
				bboxes[idx].push_back(ptr[x]);
			}
			ptr += out.w;
		}
	}
	
	std::vector<ObjDet::ObjDetItem> objs;
	for (int i = 0; i < bboxes.size(); i++) {
		int anchor_idx = i % 7;

		int idx = (int)i / 7;
		int w_idx = idx % out.w;
		int h_idx = (int)idx / out.w;

		float x = ((float)w_idx + sigmoid(bboxes[i][0])) / (float)out.w;
		float y = ((float)h_idx + sigmoid(bboxes[i][1])) / (float)out.h;
		float w = exp(bboxes[i][2]) * anchors[anchor_idx][0] / (float)out.w;
		float h = exp(bboxes[i][3]) * anchors[anchor_idx][1] / (float)out.h;

		float score = sigmoid(bboxes[i][4]);
		/*printf("%f, %f, %f, %f, %f", x, y, w, h, score);
		printf("\n");*/

		std::vector<float> cls_score(bboxes[i].begin()+5, bboxes[i].end());
		cls_score = softmax(cls_score);
		ObjDet::ClassType cls_idx = (ObjDet::ClassType) argmax(cls_score.begin(), cls_score.end());

		ObjDet::ObjDetItem obj;
		obj.box = ObjDet::BoundingBox(ObjDet::BoxFormat::CXCYHW, x, y, w, h);
		obj.score = score;
		obj.classtype = cls_idx;

		objs.push_back(obj);
		//for (const float& k : cls_score)
		//	std::cout << k << " ";
		//std::cout << std::endl;
		//printf("\n");
	}

	objs = nms_boxes(objs, 0.4f, 0.5f);

	/*std::vector<ncnn::Blob> aa = mynet.blobs;*/

	return objs;
}


int main(int argc, char** argv)
{
	std::vector<cv::String> files;
	cv::glob("F:\\\\dataset\\\\images\\\\*.jpg", files, true);

	clock_t start, finish;
	for (int i = 0; i < files.size(); i++) {
		cv::Mat img = cv::imread(files[i], 1);
		if (img.empty())
		{
			fprintf(stderr, "cv::imread %s failed\n", "d:a.png");
			return -1;
		}
		else
		{
			/*cv::imshow("raw", m);
			cv::waitKey(0);*/
		}

		std::vector<ObjDet::ObjDetItem> bboxes;
		double totaltime;
		
		// inference
		start = clock();
		bboxes = detect(img);
		finish = clock();
		totaltime = (double)(finish - start);
		std::cout << "runing time:" << totaltime << "ms" << std::endl;

		// visualize the results
		draw_objects(img, bboxes);
		
	}

	cv::waitKey(0);
	/*print_topk(cls_scores, 3);*/
	return 0;
}