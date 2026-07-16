import logging

from datetime import datetime
from typing import Dict, Any, List, Tuple

from core.data.databuses import DataBus
from core.data.simulation_flow import SimulationFlow
from core.process.process_node import ProcessNode
from core.orchestrator.orchestrator_state import OrchestratorState
from core.orchestrator.result_manager import ResultManager
from core.orchestrator.influent_initializer import InfluentInitializer
from core.connection.connection_manager import ConnectionManager
from core.connection.connection import Connection

class SimulationOrchestrator:
    """Cerveau du simulateur - coordination entre procédés et flux"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        sim_config = config.get('simulation', {})
        self.logger = logging.getLogger(f"{__name__}.{config.get('name','sim')}")

        self.state = OrchestratorState(
            start_time=datetime.fromisoformat(sim_config.get('start_time')),
            end_time=datetime.fromisoformat(sim_config.get('end_time')),
            timestep_hours=sim_config.get('timestep_hours', 0.1)
        )
        self.databus = DataBus()
        self.simulation_flow = SimulationFlow()
        self.result_manager = ResultManager(self.simulation_flow)

        self.process_nodes: List[ProcessNode] = []
        self.process_map: Dict[str, ProcessNode] = {}

        self.connection_manager = ConnectionManager()

        self.is_running = False

    def add_process(self, process: ProcessNode) -> None:
        """
        Ajoute un ProcessNode à la chaîne de simulation

        Args:
            process (ProcessNode): Instance de ProcessNode à ajouter
        """
        if process.node_id in self.process_map:
            raise ValueError(f"Duplicate ProcessNode ID '{process.node_id}'")
        self.process_nodes.append(process)
        self.process_map[process.node_id] = process
        self.logger.info(f"ProcessNode ajouté : {process.name}")

    def initialize(self) -> None:
        """
        Initialise tous les processNodes et prépare la simulation
        """
        self.logger.info("Initialisation de la simulation ...")
        self._setup_connections()
        for process in self.process_nodes:
            process.initialize()
        influent = InfluentInitializer.create_from_config(self.config, self.state.current_time)
        self.databus.write_flow('influent', influent)
        self.simulation_flow.add_flow('influent', influent)
        self.logger.info("Simulation initialisée")

    def _setup_connections(self) -> None:
        """Configure le gestaionnaire de connexions depuis la config"""
        connections_config = self.config.get('connections', {})

        if not connections_config:
            self.logger.warning("Aucune connexion définie, création d'une chaîne séquentielle")
            self._create_sequential_connections()
            return
        
        for conn in connections_config:
            self.connection_manager.add_connection(
                source_id=conn['source'],
                target_id=conn['target'],
                flow_fraction=conn.get('fraction', 1.0),
                is_recycle=conn.get('is_recycle', False),
                source_port=conn.get('source_port', 'main')
            )
        
        validation = self.connection_manager.validate()
        if validation['errors']:
            for error in validation['errors']:
                self.logger.error(error)
            raise ValueError("Erreurs dans le graphe de connexions")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                self.logger.warning(warning)

        self.logger.info("\n"+self.connection_manager.visualize_ascii())
   
    def _create_sequential_connections(self) -> None:
        """Crée une chaîne séquentielle simple"""
        if not self.process_nodes:
            return
        
        first = self.process_nodes[0]
        self.connection_manager.add_connection('influent', first.node_id, 1.0, False)

        for i in range(len(self.process_nodes) - 1):
            current = self.process_nodes[i]
            next_proc = self.process_nodes[i+1]
            self.connection_manager.add_connection(
                current.node_id,
                next_proc.node_id,
                1.0,
                False
            )

    def run(self) -> Dict[str, Any]:
        """
        Exécute la simulation complète

        Returns:
            Dict[str, Any]: Dictionnaire contenant les résultats de la simulation
        """
        self.is_running = True
        total_steps = self.state.total_steps
        self.logger.info(f"Simulation : {total_steps} pas de temps")

        while self.state.current_time < self.state.end_time:
            self._run_timestep()
            self.state.advance()
            if self.state.current_step % 100 == 0:
                self.logger.info(f"{self.state.progress_percent():.1f}% complété")
        self.is_running = False
        metadata = {
            'sim_name': self.config.get('name', 'simulation'),
            'start_time': str(self.state.start_time),
            'end_time': str(self.state.end_time),
            'total_hours': (self.state.end_time - self.state.start_time).total_seconds()/3600,
            'timestep': self.state.timestep,
            'steps_completed': self.state.current_step
        }
        return self.result_manager.collect(metadata)

    def _run_timestep(self) -> None:
        """
        Exécute un seul pas de temps de simulation
        """
        execution_order = self.connection_manager.get_execution_order()

        for node_id in execution_order:
            if node_id == 'influent':
                continue

            process = self.process_map.get(node_id)
            if not process:
                continue

            inputs = self._get_process_inputs(process)

            outputs = process.process(inputs, dt=self.state.timestep)
            process.update_state(outputs)

            flow = self._create_output_flow(process, outputs)
            self.databus.write_flow(process.node_id, flow)
            self.simulation_flow.add_flow(process.node_id, flow)

    def _resolve_source_view(self, source_flow, connection: Connection) -> Tuple[float, Dict[str, float]]:
        """
        Résout (débit de base, composants) pour une connexion donnée, selon le
        port source demandé.

        Le port "underflow" pioche dans le flux de boues concentrées exposé par
        certains procédés (ex. décanteur) plutôt que dans le flux principal
        (overflow/effluent clarifié), afin d'éviter qu'un procédé en aval
        (ex. digesteur) ne reçoive par erreur une fraction du flux clarifié.

        Args:
            source_flow (FlowData): Flux source lu depuis le DataBus
            connection (Connection): Connexion décrivant le port et la fraction

        Returns:
            Tuple[float, Dict[str, float]]: (débit total du port, composants du port)
        """
        if connection.source_port == 'underflow':
            if not source_flow.underflow:
                self.logger.warning(
                    f"Connexion {connection} demande le port 'underflow' mais "
                    f"'{connection.source_id}' n'expose aucun flux underflow ; "
                    f"utilisation du flux principal en repli"
                )
            else:
                underflow = source_flow.underflow
                return underflow.get('flowrate', 0.0), underflow.get('components', {})
        return source_flow.flowrate, source_flow.components

    def _get_process_inputs(self, process: ProcessNode) -> Dict[str, Any]:
        """
        Récupère les inputs pour un ProcessNode depuis le DataBus

        Args:
            process (ProcessNode): ProcessNode dont on veut les inputs

        Returns:
            Dict[str, Any]: Dictionnaire des inputs
        """
        upstream_connections = self.connection_manager.get_upstream_nodes(process.node_id)

        if not upstream_connections:
            self.logger.warning(f"Aucun upstream pour {process.node_id}")
            return {}
        
        if len(upstream_connections) == 1:
            source_id, connection = upstream_connections[0]
            source_flow = self.databus.read_flow(source_id)

            if not source_flow:
                if connection.is_recycle:
                    self.logger.debug(f"Flux de recyclage non disponible pour {source_id} (normal au 1er pas)")
                else:
                    self.logger.warning(f"Flux manquant pour {source_id}")
                return {}
            
            fraction = connection.flow_fraction
            base_flowrate, base_components = self._resolve_source_view(source_flow, connection)

            # Pour un port underflow, construire un FlowData avec les concentrations
            # réelles des boues concentrées (TSS/COD underflow >> overflow).
            flow_for_input = self._build_flow_for_input(source_flow, connection)

            return {
                'flow': flow_for_input,
                'flowrate': base_flowrate * fraction,
                'temperature': source_flow.temperature,
                'components': base_components.copy()
            }
        return self._mix_multiple_sources(upstream_connections)

    def _build_flow_for_input(self, source_flow, connection: Connection):
        """
        Retourne un FlowData dont les attributs mesurables (cod, tss…) reflètent
        le port réellement utilisé.

        Pour le port 'underflow', les concentrations TSS/COD de l'overflow principal
        sont remplacées par celles des boues concentrées, afin que la fractionation
        ADM1 s'appuie sur les bonnes valeurs substrat.
        """
        if connection.source_port != 'underflow' or not source_flow.underflow:
            return source_flow

        from core.data.flow_data import FlowData
        uf = source_flow.underflow
        uf_tss = uf.get('tss', 0.0)
        # Estimation COD boues : iCOD_VSS ≈ 1.42 g COD / g VSS pour biomasse activée
        uf_cod = uf_tss * 1.42

        flow = FlowData(
            timestamp=source_flow.timestamp,
            flowrate=uf.get('flowrate', source_flow.flowrate),
            temperature=source_flow.temperature,
            tss=uf_tss,
            cod=uf_cod,
            tkn=source_flow.tkn,
            nh4=source_flow.nh4,
            no3=source_flow.no3,
            po4=source_flow.po4,
            alkalinity=source_flow.alkalinity,
            components=uf.get('components', {}).copy(),
            model_type=source_flow.model_type,
            source_node=source_flow.source_node,
        )
        return flow

    def _mix_multiple_sources(self, upstream_connections: List[Tuple[str, Connection]]) -> Dict[str, Any]:
        """
        Mélange plusiseurs flux sources avec leurs fractions respectives
        """
        total_flowrate = 0.0
        weighted_temp = 0.0
        weighted_components: Dict[str, float] = {}

        reference_flow = None

        for source_id, connection in upstream_connections:
            source_flow = self.databus.read_flow(source_id)

            if not source_flow:
                if connection.is_recycle:
                    self.logger.debug(f"Flux de recyclage non disponible pour {source_id} (normal au 1er pas)")
                else:
                    self.logger.warning(f"Flux manquant pour {source_id}")
                continue

            if reference_flow is None:
                reference_flow = source_flow

            base_flowrate, base_components = self._resolve_source_view(source_flow, connection)
            fractional_flowrate = base_flowrate * connection.flow_fraction
            total_flowrate += fractional_flowrate

            weighted_temp += source_flow.temperature * fractional_flowrate

            for component, concentration in base_components.items():
                if not isinstance(concentration, (int, float)):
                    continue
                if component not in weighted_components:
                    weighted_components[component] = 0.0
                weighted_components[component] += concentration * fractional_flowrate

        if total_flowrate == 0:
            self.logger.error("Débit total nul après mélange")
            return {}
        
        mixed_temperature = weighted_temp / total_flowrate
        mixed_components = {
            comp: value / total_flowrate
            for comp, value in weighted_components.items()
        }

        return {
            'flow': reference_flow,
            'flowrate': total_flowrate,
            'temperature': mixed_temperature,
            'components': mixed_components
        }

    def _create_output_flow(self, process: ProcessNode, outputs: Dict[str, Any]):
        """
        Crée un FlowData à partir des outputs d'un ProcessNode

        Args:
            process (ProcessNode): ProcessNode source
            outputs (Dict[str, Any]): Dictionnaire des outputs

        Returns:
            FlowData: FlowData contruit
        """
        from core.data.flow_data import FlowData
        flow = FlowData(
            timestamp=self.state.current_time,
            flowrate=outputs.get('flowrate', 0.0),
            temperature=outputs.get('temperature', 20.0),
            model_type=outputs.get('model_type'),
            source_node=process.node_id
        )

        flow.components = outputs.get('components', {}).copy()
        flow.underflow = outputs.get('underflow', {}).copy() if outputs.get('underflow') else {}

        # Séparer les métriques opérationnelles des composants chimiques :
        # - flow.components  → variables d'état ASM (si, ss, xi, xs, xbh…)
        # - flow.metrics     → métriques calculées (srt_days, svi, energy_kwh…)
        _structural_keys = {'components', 'underflow', 'flowrate', 'temperature', 'model_type'}
        for key, value in outputs.items():
            if key not in _structural_keys and key not in flow.components:
                flow.metrics[key] = value

        for key in ['cod', 'tss', 'bod', 'tkn', 'nh4', 'no3', 'po4']:
            if key in outputs:
                setattr(flow, key, outputs[key])
        return flow