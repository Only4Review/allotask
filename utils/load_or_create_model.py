import torch
import meta_learners
import CONSTANTS as see


def load_or_create_model(model_name, model_checkpoint=None, device=None, *args, **kwargs):
    if model_checkpoint:
        try:
            model = torch.load(model_checkpoint, map_location=device)
            see.logs.write("Load trained models\n")
            return model
        except Exception:
            see.logs.write("\nFailed to load model from checkpoint. Recreating from params.\n")

    modelclass = getattr(meta_learners, model_name)
    model = modelclass(*args, **kwargs)
    # assign gpu or cpu
    model = model.to(device)
    return model