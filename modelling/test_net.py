import os
import json
import argparse
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
from AnnotateHelper import annoatate_mask_bgr, mask2box
import numpy as np


def conf_model(args):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = args.exp_folder
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.MASK_ON = args.mask_on
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')  # path to the model we just trained

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.test_threshold  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    return predictor


def inference(args):
    predictor = conf_model(args)
    info_dict = {'image_id': list(),
                 'cls_pred': list(),
                 'cls_prob': list(),
                 'set': list(),
                 'label': list(),
                 'patient_id': list(),
                 }

    json_dict = {}
    data_row = {
        'set': 'test',
        'image_path': args.image_path,
        'classification': '2r3_no',
        'patient_id': 0
    }
    prediction_dict = {'image_id': args.image_path.split('.')[0],
                       'set': data_row['set'],
                       'label': data_row['classification'],
                       'patient_id': data_row['patient_id']}
    json_dict['image_id'] = args.image_path.split('.')[0]
    json_dict['patient_id'] = str(data_row['patient_id'])
    info_dict['image_id'].append(args.image_path.split('.')[0])
    info_dict['set'].append(data_row['set'])
    info_dict['label'].append(data_row['classification'])
    info_dict['patient_id'].append(data_row['patient_id'])

    im = cv2.imread(args.image_path)
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(im)
    num_predictions = len(outputs['instances'].pred_classes)
    # note that 0 means nothing was detected, even ulna and radius, this is most likely an image being rotated more
    # than +-45 degrees.
    if num_predictions in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]:
        # for ulna and radius prediction:
        for cls_name in ['radius']:
            predictions = [[cls.item(), cls_prob.item(), ins_idx] for cls, cls_prob, ins_idx in
                           zip(outputs['instances'].pred_classes, outputs['instances'].scores,
                               range(len(outputs['instances'])))]

            if len(predictions) == 1:
                ins_idx = predictions[0][2]
                bbox = outputs['instances'].pred_boxes.tensor[int(ins_idx)].cpu().numpy().tolist()
                prediction_dict['{}_pred_bbox'.format(cls_name)] = bbox
                mask = outputs['instances'].pred_masks[int(ins_idx)].cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                prediction_dict['{}_pred_segmentation'.format(cls_name)] = contours
                json_dict['bbox'] = contours[0].tolist()
                json_dict['mask'] = mask.tolist()
            elif len(predictions) > 1:
                print('{} has more than 1 prediction for {}'.format(data_row['image_path'], cls_name))
                ins_idx = predictions[0][2]
                bbox = outputs['instances'].pred_boxes.tensor[int(ins_idx)].cpu().numpy().tolist()
                prediction_dict['{}_pred_bbox'.format(cls_name)] = bbox
                mask = outputs['instances'].pred_masks[int(ins_idx)].cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                prediction_dict['{}_pred_segmentation'.format(cls_name)] = contours
                json_dict['bbox'] = contours[0].tolist()
                json_dict['mask'] = mask.tolist()
            elif len(predictions) == 0:
                print('{} has no {} prediction'.format(0, cls_name))
        # predictions other than ulna and radius
        predictions = [[cls.item(), cls_prob.item(), ins_idx] for cls, cls_prob, ins_idx in
                       zip(outputs['instances'].pred_classes, outputs['instances'].scores,
                           range(len(outputs['instances'])))]
        predictions = np.array(predictions)

        if predictions.shape[0] == 0:
            nofracture_prob = 0.7
            info_dict['cls_pred'].append('2r3_no')
            info_dict['cls_prob'].append(nofracture_prob)
            json_dict['cls_pred'] = '2r3_no'
            json_dict['cls_prob'] = nofracture_prob
            prediction_dict['cls_pred'] = '2r3_no'
            prediction_dict['cls_prob'] = nofracture_prob
            prediction_dict['pred_bbox'] = None
            prediction_dict['pred_segmentation'] = None
        else:
            cls_idx, cls_prob, ins_idx = predictions[predictions[:, 1].argmax()]
            json_dict['cls_pred'] = '2r3_yes'
            json_dict['cls_prob'] = cls_prob
            prediction_dict['cls_prob'] = cls_prob
            bbox = outputs['instances'].pred_boxes.tensor[int(ins_idx)].cpu().numpy().tolist()
            prediction_dict['pred_bbox'] = bbox
            mask = outputs['instances'].pred_masks[int(ins_idx)].cpu().numpy().astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            prediction_dict['pred_segmentation'] = contours

    else:
        raise ValueError('Multiple Fracture Predictions: {}'.format(num_predictions))


    if 'mask' in json_dict.keys():
        # start
        image_save_id = str(json_dict['image_id'])
        patient_save_id = str(json_dict['patient_id'])
        save_dir = os.path.dirname(args.json_file)

        data_save_dir = os.path.join(save_dir, f'{image_save_id}', patient_save_id)
        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)

        json_dict['cls_pred'] = 'drf_fracture' if json_dict['cls_pred'] == '2r3_yes' else 'no_fracture'

        mask_image = np.asarray(json_dict['mask']).astype(np.uint8)
        mask_image[mask_image > 0] = 255
        data_mask_save_path = os.path.join(data_save_dir, 'mask.jpg')
        cv2.imwrite(data_mask_save_path, mask_image.astype(np.uint8))
        json_dict['mask'] = data_mask_save_path

        # save_annotate
        annotate_image = annoatate_mask_bgr(im, mask_image, (0, 255, 0), 0.5).astype(np.uint8)
        box_x, box_y, box_w, box_h = mask2box(mask_image).tolist()[0][:4]
        annotate_str = "{cls_pred}_{cls_prob:.2f}".format(**json_dict)
        (text_w, text_h), _ = cv2.getTextSize(annotate_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_size = annotate_image.shape[1] * 0.25 / text_w
        cv2.putText(annotate_image, annotate_str, (box_x, box_y), cv2.FONT_HERSHEY_SIMPLEX,text_size, (255, 0, 0), 1)
        data_annotate_save_path = os.path.join(data_save_dir, 'image.jpg')
        cv2.imwrite(data_annotate_save_path, annotate_image)

        # save json
        data_json_save_path = os.path.join(data_save_dir, f'{os.path.basename(args.json_file)}')
        with open(data_json_save_path, 'w') as f:
            json.dump(json_dict, f)
        # end

    im_path = os.path.join(args.image_out_path, '{}.jpg'.format(1))
    cv2.imwrite(im_path, im)


def main(args):
    args.cls_classes = ['2r3a', '2r3b', '2r3c', '2r3_no']
    args.json_file = os.path.join(args.exp_folder, 'result.json')
    args.color_profile = {
        "ulna": {'annotation': [255, 0, 0], 'prediction': [255, 0, 0]},
        "radius": {'annotation': [255, 128, 0], 'prediction': [255, 128, 0]},
        "2r3a": {'annotation': [0, 255, 255], 'prediction': [255, 0, 255]},
        "2r3b": {'annotation': [0, 255, 255], 'prediction': [255, 0, 255]},
        "2r3c": {'annotation': [0, 255, 255], 'prediction': [255, 0, 255]},
    }

    result_df_path = os.path.join(args.exp_folder, 'result.csv')
    study_df_path = os.path.join(args.exp_folder, 'study.csv')
    args.image_out_path = os.path.join(args.exp_folder, 'generated_images')

    os.makedirs(args.image_out_path, exist_ok=True)

    if args.recompute_result or (not os.path.exists(result_df_path)) or (not os.path.exists(study_df_path)):
        inference(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_folder', type=str, default='./result')
    parser.add_argument('--config_file', type=str, default='configs/faster_rcnn_R_50_FPN_1x.yaml')
    parser.add_argument('--recompute_result', type=bool, default=True)
    parser.add_argument('--generate_image', type=bool, default=True)
    parser.add_argument('--mask_on', type=bool, default=True)
    parser.add_argument('--test_threshold', type=float, default=0.5)
    parser.add_argument('--image_path', type=str)
    args = parser.parse_args()

    print(args)

    main(args)
