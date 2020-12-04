import logging

from .histo_eval import do_ringcell_evaluation, do_monuseg_evaluation, eval_detection_monuseg, eval_detection_ringcell, calc_froc


def ringcell_evaluation(dataset, predictions, output_folder, box_only, **args):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("histo evaluation doesn't support box_only, ignored.")
    logger.info("performing histo evaluation, ignored iou_types.")
    return do_ringcell_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        dataset_nr=args["dataset_nr"],
        predictions_nr=args["predictions_nr"],
    )

def monuseg_evaluation(dataset, predictions, output_folder, box_only, **args):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("histo evaluation doesn't support box_only, ignored.")
    logger.info("performing histo evaluation, ignored iou_types.")
    return do_monuseg_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger
    )
