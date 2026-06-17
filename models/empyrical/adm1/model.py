import numpy as np
from typing import Dict, Optional, List
from models.reaction_model import ReactionModel
from core.model.model_registry import ModelRegistry
from models.empyrical.adm1.kinetics import calculate_process_rates
from models.empyrical.adm1.stoichiometry import build_stoichiometric_matrix


class ADM1Model(ReactionModel):

    N_LIQUID = 26 # nombre de composants dans la phase liquide

    def __init__(self, params: Optional[Dict[str, float]] = None):
        super().__init__(params)
        registry = ModelRegistry.get_instance()
        model_def = registry.get_model_definition("ADM1Model")
        self.DEFAULT_PARAMS = model_def.get_default_params()
        self.COMPONENT_INDICES = {
            name: i for i, name in enumerate(model_def.get_components_names())
        }
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._S = None

    @property
    def model_type(self) -> str:
        return "ADM1Model"

    def get_component_names(self) -> list:
        return list(self.COMPONENT_INDICES.keys())

    def process_rates(self, concentrations: np.ndarray) -> np.ndarray:
        return calculate_process_rates(concentrations, self.params)

    def stoichiometric_matrix(self) -> np.ndarray:
        if self._S is None:
            self._S = build_stoichiometric_matrix(self.params)
        return self._S

    def concentrations_to_dict(self, state: np.ndarray) -> Dict[str, float]:
        return {name: state[i] for name, i in self.COMPONENT_INDICES.items()}

    def dict_to_concentrations(self, state_dict: Dict[str, float]) -> np.ndarray:
        c = np.zeros(self.N_LIQUID)
        for name, value in state_dict.items():
            if name in self.COMPONENT_INDICES:
                c[self.COMPONENT_INDICES[name]] = value
        return c







