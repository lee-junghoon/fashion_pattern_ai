from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# GT와 prediction 불러오기
coco_gt = COCO("annotation.json")
coco_dt = coco_gt.loadRes("results/predictions_coco_format.json")

# 평가 실행
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
