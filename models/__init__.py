from models.mlp_predictor import MLPPredictor
from models.transformer_predictor import TransformerPredictor
from models.identity_head import IdentityHead, DualHeadModel

MODEL_REGISTRY = {
    "mlp": MLPPredictor,
    "transformer": TransformerPredictor,
}
