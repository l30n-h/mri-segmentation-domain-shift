from nnunet.evaluation.evaluator import NiftiEvaluator
from nnunet.evaluation.metrics import ALL_METRICS
import surface_distance.metrics as surf_dc

def SDice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, sdice_tolerance=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    surface_distances = surf_dc.compute_surface_distances(test, reference, voxel_spacing)
    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, sdice_tolerance)

ALL_METRICS['SDice'] = SDice

class SDiceNiftiEvaluator(NiftiEvaluator):
    def __init__(self, *args, **kwargs):
        super(SDiceNiftiEvaluator, self).__init__(*args, **kwargs)
        self.add_metric("SDice")