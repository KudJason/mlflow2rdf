from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD
from datetime import datetime, timezone
import math
import re

from ..config import (
    MLS, MLSO, MLSEA, DCTERMS, PROV, FOAF, ADMS, SKOS,
    MLFLOW, MLST_TASK, MLSO_EVAL, MLSO_DC,
    DATASET_GRAPH_URI, RUNS_GRAPH_URI, slugify
)

from ..config_loader import load_config

# =============================================================================
# 声明式配置加载
# =============================================================================

# Load YAML configs once at module level.
# These replace the previously hardcoded dicts.
_cfg = load_config()

# Property routing: param key → entity domain (model/run/algorithm/data)
_routing = _cfg.property_routing  # Dict[str, list]
PARADIGM_PROPERTY_CONFIG = {k: set(v) for k, v in _routing.items()}

# Property URI map: param key → CamelCase MLSO property suffix
PROPERTY_URI_MAP = dict(_cfg.property_uris)

# Paradigm label map: tag value → display label
PARADIGM_LABEL_MAP = dict(_cfg.paradigm_labels)


def to_pascal_case(text: str) -> str:
    """将 kebab-case/snake_case 转为 PascalCase (如 'self-supervised-learning' → 'SelfSupervisedLearning')"""
    return ''.join(part.capitalize() for part in text.replace('-', '_').split('_'))


# =============================================================================
# Metric 类型映射 (从 YAML 配置动态加载)
# =============================================================================

def _build_metric_type_map() -> dict:
    """Build METRIC_TYPE_MAP from YAML config, resolving URI strings to URIRefs."""
    raw = _cfg.metric_types  # Dict[str, str]
    result = {}
    # "MLS.EvaluationMeasure" → MLS.EvaluationMeasure (generic fallback)
    generic = 'MLS.EvaluationMeasure'
    for key, val in raw.items():
        if val == generic:
            result[key] = MLS.EvaluationMeasure
        else:
            # val is a suffix like 'root_mean_squared_error', 'accuracy', etc.
            # Resolve as attribute of MLSO_EVAL namespace
            result[key] = getattr(MLSO_EVAL, val, MLS.EvaluationMeasure)
    return result


METRIC_TYPE_MAP = _build_metric_type_map()


# =============================================================================
# Evaluation Measure → OpenML URI Mapping (从 YAML 配置加载)
# =============================================================================

MEASURE_OPENML_MAP = dict(_cfg.task_types.get('openml_measures', {}))


# =============================================================================
# Task Type → MLST Task Type Taxonomy (从 YAML 配置加载)
# =============================================================================

def _build_task_type_map() -> dict:
    """Build TASK_TYPE_MAP from YAML config, constructing MLST_TASK URIRefs."""
    raw = _cfg.task_types.get('task_types', {})  # Dict[str, str]
    result = {}
    for key, task_name in raw.items():
        # task_name is PascalCase like 'Classification', 'GenerativeModeling'
        result[key] = MLST_TASK[task_name]
    return result


TASK_TYPE_MAP = _build_task_type_map()


def get_metric_type_uri(metric_name: str) -> URIRef:
    """Returns the ontology type URI for a given metric name."""
    metric_lower = metric_name.lower()
    if metric_lower in METRIC_TYPE_MAP:
        return METRIC_TYPE_MAP[metric_lower]
    return MLS.EvaluationMeasure


def infer_artifact_type(artifact_path: str) -> URIRef:
    """Infers the artifact type URI based on the file path/name."""
    path_lower = artifact_path.lower()

    if 'confusion_matrix' in path_lower:
        return MLSO.ConfusionMatrix
    if 'roc_curve' in path_lower or 'precision_recall' in path_lower:
        return MLSO.ROCCurve
    if artifact_path.endswith(('.png', '.jpg', '.jpeg')):
        return MLSO.ImageArtifact
    if 'tfidf_vectorizer' in path_lower or 'vectorizer' in path_lower:
        return MLSO.TextVectorizer
    if 'tokenizer' in path_lower:
        return MLSO.Tokenizer
    if 'feature_importance' in path_lower:
        return MLSO.FeatureImportance
    if artifact_path.endswith('.json') and ('config' in path_lower or 'anchor' in path_lower):
        return MLSO.ConfigurationArtifact
    if artifact_path.endswith(('.csv', '.json')):
        return MLSO.TabularArtifact
    if artifact_path.endswith(('.pt', '.bin', '.pth')):
        return MLSO.ModelArtifact
    if 'model' in path_lower and artifact_path.endswith(('.pkl', '.joblib')):
        return MLSO.ModelArtifact
    return MLSO.Artifact


def _get_property_uri(param_key: str) -> str:
    """将 snake_case/kebab-case 参数名转换为 MLSO 属性名"""
    normalized = param_key.lower().replace('-', '_')

    # 先查映射表
    if normalized in PROPERTY_URI_MAP:
        return PROPERTY_URI_MAP[normalized]

    # 动态转换：lora_rank → hasLoraRank
    parts = normalized.split('_')
    if len(parts) >= 2:
        camel = ''.join(p.capitalize() for p in parts)
        return f'has{camel}'

    return f'has{normalized.capitalize()}'


def _normalize_param_key(param_key: str) -> str:
    """将参数名标准化为 snake_case"""
    return param_key.lower().replace('-', '_')


def literal_from_value(value):
    """Converts a value to an appropriate RDF Literal."""
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, int):
            return Literal(value, datatype=XSD.integer)
        elif isinstance(value, float):
            if math.isfinite(value):
                return Literal(value, datatype=XSD.float)
            else:
                return None

    if isinstance(value, bool):
        return Literal(value, datatype=XSD.boolean)

    text = str(value).strip()
    if text == '' or text.lower() == 'none':
        return None

    if text.lower() in {'true', 'false'}:
        return Literal(text.lower() == 'true', datatype=XSD.boolean)

    try:
        if '.' in text:
            number = float(text)
            if math.isfinite(number):
                return Literal(number, datatype=XSD.float)
        else:
            return Literal(int(text), datatype=XSD.integer)
    except ValueError:
        pass

    return Literal(text)


def map_run_status(status_val):
    """Maps MLflow status to ADMS status URI."""
    status_map = {
        'FINISHED': ADMS.Completed,
        'FAILED': ADMS.Failed,
        'RUNNING': ADMS.UnderDevelopment,
        'SCHEDULED': ADMS.UnderDevelopment,
        'KILLED': ADMS.Deprecated
    }
    return status_map.get(status_val, ADMS.Unknown)


def build_dataset_graph(records: list[dict]) -> tuple[Graph, dict, dict]:
    """Generates the base Dataset graph from MLflow records."""
    graph = Graph(identifier=URIRef(DATASET_GRAPH_URI))
    graph.bind('mls', MLS); graph.bind('mlsea', MLSEA); graph.bind('dcterms', DCTERMS)

    dataset_uri_map = {}
    processing_uri_map = {}

    for record in records:
        dataset_name = record['tags'].get('mlsea.dataset') or 'unknown-dataset'
        dataset_slug = slugify(dataset_name)
        dataset_uri = dataset_uri_map.setdefault(dataset_name, MLFLOW[f'dataset/{dataset_slug}'])

        if (dataset_uri, RDF.type, MLS.Dataset) not in graph:
            graph.add((dataset_uri, RDF.type, MLS.Dataset))
            graph.add((dataset_uri, DCTERMS.title, Literal(dataset_name)))
            graph.add((dataset_uri, MLSO.hasDatasetSource, Literal(record['tags'].get('mlsea.dataset_source', 'kaggle'), datatype=XSD.string)))
            graph.add((dataset_uri, RDF.type, MLS.Data))

            fallback = _cfg.task_types.get('known_datasets', {}).get(dataset_slug, {})

            rows = record['params'].get('rows') or fallback.get('rowCount')
            if rows:
                graph.add((dataset_uri, MLSO.hasRowCount, literal_from_value(rows)))
                # DatasetCharacteristic node — aligns with OpenML data model
                char_uri = URIRef(f"{str(dataset_uri)}/characteristic/number_of_instances")
                graph.add((char_uri, RDF.type, MLS.DatasetCharacteristic))
                graph.add((char_uri, MLSO.hasDataCharacteristicType,
                           MLSO_DC.number_of_instances))
                graph.add((char_uri, MLS.hasValue, literal_from_value(rows)))

            feats = record['params'].get('feature_count') or fallback.get('featureCount')
            if feats:
                graph.add((dataset_uri, MLSO.hasFeatureCount, literal_from_value(feats)))
                # DatasetCharacteristic node — aligns with OpenML data model
                char_uri = URIRef(f"{str(dataset_uri)}/characteristic/number_of_features")
                graph.add((char_uri, RDF.type, MLS.DatasetCharacteristic))
                graph.add((char_uri, MLSO.hasDataCharacteristicType,
                           MLSO_DC.number_of_features))
                graph.add((char_uri, MLS.hasValue, literal_from_value(feats)))

            target = record['params'].get('target') or fallback.get('targetName')
            if target:
                graph.add((dataset_uri, MLSO.hasTargetFeature, Literal(target, datatype=XSD.string)))

        strategy = record['params'].get('processing_strategy')
        if strategy:
            strategy_slug = slugify(strategy)
            strategy_uri = processing_uri_map.setdefault(strategy, MLFLOW[f'processing/{strategy_slug}'])
            if (strategy_uri, RDF.type, MLS.DataTransformation) not in graph:
                graph.add((strategy_uri, RDF.type, MLS.DataTransformation))
                graph.add((strategy_uri, RDFS.label, Literal(strategy)))
                note = record['tags'].get('processing_notes')
                if note:
                    graph.add((strategy_uri, DCTERMS.description, Literal(note)))

            graph.add((dataset_uri, MLS.hasPart, strategy_uri))

        # Data 级参数处理（如 chunk_size for RAG）
        for p_key, p_val in record['params'].items():
            p_norm = _normalize_param_key(p_key)
            if p_norm in PARADIGM_PROPERTY_CONFIG['data']:
                lit = literal_from_value(p_val)
                if lit is not None:
                    prop_uri = MLSO[_get_property_uri(p_key)]
                    graph.add((dataset_uri, prop_uri, lit))

    return graph, dataset_uri_map, processing_uri_map




def build_runs_graph_skeleton(records: list[dict], dataset_uris: dict, processing_uris: dict) -> Graph:
    """
    Builds the runs graph with complete MLSO/MLST support.

    Generates:
    - mls:Run with all metadata
    - mls:Algorithm with paradigm-specific properties
    - mls:Model with LoRA/Adapter properties
    - Pipeline relationships (distilledFrom, basedOnModel, wasPretrainedFrom, etc.)
    """
    graph = Graph(identifier=URIRef(RUNS_GRAPH_URI))
    graph.bind('mls', MLS); graph.bind('mlso', MLSO); graph.bind('mlsea', MLSEA)
    graph.bind('dcterms', DCTERMS); graph.bind('prov', PROV); graph.bind('foaf', FOAF)
    graph.bind('adms', ADMS); graph.bind('skos', SKOS)

    task_uri_map = {}
    algo_uri_map = {}

    for record in records:
        run_id = record['run_id']
        run_uri = MLFLOW[f'run/{run_id}']

        # === Run Base Properties ===
        graph.add((run_uri, RDF.type, MLS.Run))
        graph.add((run_uri, DCTERMS.identifier, Literal(run_id)))

        if record['status']:
            graph.add((run_uri, ADMS.status, map_run_status(record['status'])))

        if record['source_version']:
            graph.add((run_uri, DCTERMS.hasVersion, Literal(record['source_version'], datatype=XSD.string)))

        entry_point = record.get('entry_point_name')
        if entry_point:
            graph.add((run_uri, MLSO.hasEntryPoint, Literal(entry_point, datatype=XSD.string)))

        lifecycle_stage = record.get('lifecycle_stage')
        if lifecycle_stage:
            graph.add((run_uri, MLSO.hasLifecycleStage, Literal(lifecycle_stage, datatype=XSD.string)))

        run_name = record['run_name'] or record['tags'].get('mlflow.runName')
        if run_name:
            graph.add((run_uri, RDFS.label, Literal(run_name)))

        # === Dataset & Processing Links ===
        dataset_name = record['tags'].get('mlsea.dataset')
        dataset_uri = None
        if dataset_name and dataset_name in dataset_uris:
            dataset_uri = dataset_uris[dataset_name]
            graph.add((run_uri, MLS.hasInput, dataset_uri))
            graph.add((run_uri, MLSO.hasDataset, dataset_uri))
            # Add modality to Dataset (ObjectProperty → DataModality SKOS Concept)
            modality_tag = record['tags'].get('mlsea.modality')
            if modality_tag:
                modality_slug = slugify(modality_tag)
                modality_uri = MLFLOW[f'concept/modality/{modality_slug}']
                graph.add((modality_uri, RDF.type, SKOS.Concept))
                graph.add((modality_uri, RDFS.label, Literal(modality_tag)))
                graph.add((modality_uri, SKOS.prefLabel, Literal(modality_tag, lang='en')))
                graph.add((dataset_uri, MLSO.hasModality, modality_uri))

        strategy = record['params'].get('processing_strategy')
        if strategy and strategy in processing_uris:
            graph.add((run_uri, MLS.hasInput, processing_uris[strategy]))

        # === Timestamps ===
        for field, predicate in [('start_time', DCTERMS.created), ('end_time', DCTERMS.modified)]:
            timestamp = record.get(field)
            if timestamp and str(timestamp).lower() not in ('null', 'none', ''):
                try:
                    dt = datetime.fromtimestamp(int(timestamp) / 1000, tz=timezone.utc)
                    graph.add((run_uri, predicate, Literal(dt.isoformat(), datatype=XSD.dateTime)))
                except (ValueError, TypeError):
                    pass

        # === User ===
        user_id = record.get('user_id')
        if user_id:
            user_uri = MLFLOW[f'user/{slugify(user_id)}']
            graph.add((user_uri, RDF.type, PROV.Agent))
            graph.add((user_uri, FOAF.name, Literal(user_id)))
            graph.add((run_uri, PROV.wasAssociatedWith, user_uri))

        # === Source Software ===
        source_name = record['tags'].get('mlflow.source.name')
        if source_name:
            source_uri = MLFLOW[f'software/{slugify(source_name)}']
            graph.add((source_uri, RDF.type, MLS.Software))
            graph.add((source_uri, RDFS.label, Literal(source_name)))
            framework = record['params'].get('sklearn_version') or record['tags'].get('mlsea.framework')
            if framework:
                graph.add((source_uri, MLSO.hasFramework, Literal(framework, datatype=XSD.string)))
            graph.add((run_uri, MLS.executes, source_uri))

        # === Experiment Link ===
        exp_id = record.get('experiment_id')
        exp_name = record.get('experiment_name')
        if exp_id:
            exp_uri = MLFLOW[f'experiment/{exp_id}']
            if (exp_uri, RDF.type, MLS.Experiment) not in graph:
                graph.add((exp_uri, RDF.type, MLS.Experiment))
                if exp_name:
                    graph.add((exp_uri, RDFS.label, Literal(exp_name)))
            graph.add((run_uri, MLS.belongsToExperiment, exp_uri))

        # === Task ===
        task_type = record['tags'].get('mlsea.task_type') or record['params'].get('task_type')
        if task_type:
            task_slug = slugify(task_type)
            mlst_concept_uri = MLST_TASK[to_pascal_case(task_slug)]
            if task_slug not in task_uri_map:
                task_uri = MLFLOW[f'task/{task_slug}']
                task_concept_uri = MLFLOW[f'concept/task-type/{task_slug}']
                graph.add((task_uri, RDF.type, MLS.Task))
                graph.add((task_uri, RDFS.label, Literal(task_type)))
                graph.add((task_concept_uri, RDF.type, SKOS.Concept))
                graph.add((task_concept_uri, RDFS.label, Literal(task_type)))
                graph.add((task_uri, MLSO.hasTaskType, task_concept_uri))
                graph.add((task_concept_uri, SKOS.prefLabel, Literal(task_type, lang='en')))
                # Link to MLSO taxonomy for cross-system alignment
                mlso_match = TASK_TYPE_MAP.get(task_slug)
                if mlso_match:
                    graph.add((task_concept_uri, SKOS.closeMatch, mlso_match))
                task_uri_map[task_slug] = task_uri
            graph.add((run_uri, MLS.executes, task_uri_map[task_slug]))
            graph.add((mlst_concept_uri, RDF.type, SKOS.Concept))
            graph.add((run_uri, MLSO.hasDownstreamTask, mlst_concept_uri))

        # === Algorithm ===
        algo_name = record['tags'].get('mlsea.algorithm')
        if not algo_name:
            algo_name = record['params'].get('model_name')
        if not algo_name and run_name and '__' in run_name:
            algo_name = run_name.split('__')[0]

        algo_uri = None
        if algo_name:
            algo_slug = slugify(algo_name)
            if algo_slug not in algo_uri_map:
                algo_uri = MLFLOW[f'algorithm/{algo_slug}']
                graph.add((algo_uri, RDF.type, MLS.Algorithm))
                graph.add((algo_uri, RDFS.label, Literal(algo_name)))

                model_family = record['tags'].get('model_family')
                if model_family:
                    family_concept = MLFLOW[f'concept/algorithm-type/{slugify(model_family)}']
                    graph.add((family_concept, RDF.type, SKOS.Concept))
                    graph.add((family_concept, RDFS.label, Literal(model_family)))
                    graph.add((algo_uri, MLSO.hasAlgorithmType, family_concept))
                    graph.add((algo_uri, MLSO.hasEstimatorName, Literal(model_family)))
                elif algo_name:
                    graph.add((algo_uri, MLSO.hasEstimatorName, Literal(algo_name)))
                paradigm = record['tags'].get('mlsea.paradigm')
                if paradigm:
                    normalized = PARADIGM_LABEL_MAP.get(paradigm, paradigm)
                    # Create paradigm SKOS concept and link via ObjectProperty
                    paradigm_slug = slugify(normalized)
                    paradigm_concept_uri = MLFLOW[f'concept/paradigm/{paradigm_slug}']
                    graph.add((paradigm_concept_uri, RDF.type, SKOS.Concept))
                    graph.add((paradigm_concept_uri, RDFS.label, Literal(normalized)))
                    graph.add((paradigm_concept_uri, SKOS.prefLabel, Literal(normalized, lang='en')))
                    graph.add((algo_uri, MLSO.hasParadigm, paradigm_concept_uri))
                algo_uri_map[algo_slug] = algo_uri
            else:
                algo_uri = algo_uri_map[algo_slug]
            graph.add((run_uri, MLS.realizes, algo_uri))

        # === Model ===
        model_uri = MLFLOW[f'model/{run_id}']
        graph.add((model_uri, RDF.type, MLS.Model))
        graph.add((model_uri, RDFS.label, Literal(f"model-{run_id[:8]}")))
        if dataset_uri:
            graph.add((model_uri, MLSO.trainedOn, dataset_uri))
        graph.add((run_uri, MLS.hasOutput, model_uri))

        # === Artifacts ===
        for artifact_path in record.get('artifacts', []):
            artifact_slug = slugify(artifact_path.replace('/', '-').replace('.', '-'))
            artifact_uri = URIRef(f"{str(run_uri)}/artifact/{artifact_slug}")
            artifact_type = infer_artifact_type(artifact_path)

            graph.add((artifact_uri, RDF.type, MLS.Artifact))
            graph.add((artifact_uri, RDF.type, artifact_type))
            graph.add((artifact_uri, RDFS.label, Literal(artifact_path)))
            graph.add((run_uri, MLS.hasOutput, artifact_uri))

        # === HyperParameterSetting + Paradigm Properties ===
        for p_key, p_val in record['params'].items():
            lit = literal_from_value(p_val)
            if lit is None:
                continue

            param_slug = slugify(p_key)
            hp_setting_uri = URIRef(f"{str(run_uri)}/param/{param_slug}")
            hp_def_uri = MLFLOW[f'hyperparameter/{param_slug}']

            graph.add((hp_setting_uri, RDF.type, MLS.HyperParameterSetting))
            graph.add((hp_setting_uri, MLS.specifiedBy, hp_def_uri))
            graph.add((hp_def_uri, RDFS.label, Literal(p_key)))
            graph.add((hp_setting_uri, MLS.hasValue, lit))
            graph.add((run_uri, MLS.hasInput, hp_setting_uri))

            # 提取 paradigm-specific 属性
            _map_paradigm_property(run_uri, model_uri, algo_uri, dataset_uri, p_key, lit, graph, record)

        # === ModelEvaluation for metrics ===
        for m_key, m_val in record['metrics'].items():
            lit = literal_from_value(m_val)
            if lit is None:
                continue

            metric_slug = slugify(m_key)
            ev_uri = URIRef(f"{str(run_uri)}/metric/{metric_slug}")
            measure_uri = MLFLOW[f'evaluationMeasure/{metric_slug}']
            metric_type_uri = get_metric_type_uri(m_key)

            graph.add((ev_uri, RDF.type, MLS.ModelEvaluation))
            graph.add((ev_uri, MLS.specifiedBy, measure_uri))
            graph.add((measure_uri, RDF.type, metric_type_uri))
            graph.add((measure_uri, RDFS.label, Literal(m_key)))
            # MLSO taxonomy link — aligns with OpenML's hasEvaluationMeasureType pattern
            if metric_type_uri != MLS.EvaluationMeasure:
                graph.add((measure_uri, MLSO.hasEvaluationMeasureType, metric_type_uri))
            # Cross-system alignment: link to OpenML evaluation measure namespace
            openml_name = MEASURE_OPENML_MAP.get(m_key.lower())
            if openml_name:
                openml_uri = URIRef(f'http://w3id.org/mlsea/openml/evaluationMeasure/{openml_name}')
                graph.add((measure_uri, SKOS.closeMatch, openml_uri))
            graph.add((ev_uri, MLS.hasValue, lit))
            graph.add((run_uri, MLS.hasOutput, ev_uri))

        # === Pipeline Relationships ===
        _add_pipeline_relationships(run_uri, model_uri, algo_uri, record, graph)

    # Post-processing: ensure all mls:Run URIs have dcterms:identifier + dcterms:created
    # Fixes placeholder URIs created from run names (e.g., mlsea.pretrain_run_id='byol_pretext')
    name_to_id = {}
    id_to_record = {}
    for r in records:
        rid = r['run_id']
        id_to_record[rid] = r
        rname = r.get('run_name') or r.get('tags', {}).get('mlflow.runName')
        if rname:
            name_to_id[rname] = rid

    for s in list(graph.subjects(RDF.type, MLS.Run)):
        has_id = bool(list(graph.objects(s, DCTERMS.identifier)))
        has_created = bool(list(graph.objects(s, DCTERMS.created)))
        if has_id and has_created:
            continue

        slug = str(s).split('/')[-1]
        matched = None

        # Try slug as run_id first
        if slug in id_to_record:
            matched = id_to_record[slug]
        # Try slug as run_name
        elif slug in name_to_id:
            matched = id_to_record.get(name_to_id[slug])

        if matched:
            if not has_id:
                graph.add((s, DCTERMS.identifier, Literal(matched['run_id'])))
            if not has_created and matched.get('start_time'):
                ts = int(matched['start_time'])
                if ts > 1_000_000_000_000:  # milliseconds → seconds
                    ts //= 1000
                dt_val = datetime.fromtimestamp(ts, tz=timezone.utc)
                graph.add((s, DCTERMS.created, Literal(dt_val, datatype=XSD.dateTime)))

    return graph


def _map_paradigm_property(run_uri: URIRef, model_uri: URIRef, algo_uri: URIRef,
                           data_uri: URIRef, param_key: str, lit: Literal,
                           graph: Graph, record: dict):
    """
    Maps paradigm-specific MLflow params to semantic MLSO predicates.

    根据属性类型添加到正确的实体：
    - Model 级: LoRA, Adapter 参数
    - Algorithm 级: Distillation, Contrastive, Multi-Modal 参数
    - Run 级: Time-Series, Self-Supervised 参数
    - Data 级: RAG chunk_size 等
    """
    p_norm = _normalize_param_key(param_key)
    params = record.get('params', {})

    # 1. Model 级属性
    if p_norm in PARADIGM_PROPERTY_CONFIG['model']:
        prop_name = _get_property_uri(param_key)
        prop_uri = MLSO[prop_name]

        # 对象属性（base_model）需要特殊处理
        if p_norm == 'base_model':
            base_model_name = str(lit)
            base_model_uri = MLFLOW[f'model/{slugify(base_model_name)}']
            graph.add((model_uri, MLSO.hasBaseModel, base_model_uri))
        else:
            graph.add((model_uri, prop_uri, lit))
        return

    # 2. Run 级属性
    if p_norm in PARADIGM_PROPERTY_CONFIG['run']:
        prop_name = _get_property_uri(param_key)
        prop_uri = MLSO[prop_name]

        # Self-supervised: learning_stage 特殊处理
        if p_norm == 'learning_stage':
            graph.add((run_uri, prop_uri, lit))
            # 如果是下游任务，记录 pretext run
            if str(lit).lower() == 'downstream':
                pretext_run_id = record.get('tags', {}).get('mlsea.pretrain_run_id')
                if pretext_run_id:
                    pretext_uri = MLFLOW[f'run/{pretext_run_id}']
                    graph.add((run_uri, MLSO.wasPretrainedFrom, pretext_uri))
        else:
            graph.add((run_uri, prop_uri, lit))
        return

    # 3. Algorithm 级属性
    if p_norm in PARADIGM_PROPERTY_CONFIG['algorithm'] and algo_uri:
        prop_name = _get_property_uri(param_key)
        prop_uri = MLSO[prop_name]

        # === Skip KD params — handled entirely in _add_pipeline_relationships ===
        if p_norm in ('teacher_model', 'student_model',
                       'distillation_temperature', 'distillation_alpha', 'distillation_loss'):
            return  # KD properties are handled in _add_pipeline_relationships

        if p_norm == 'image_encoder':
            encoder_name = str(lit)
            encoder_uri = MLFLOW[f'algorithm/{slugify(encoder_name)}']
            graph.add((encoder_uri, RDF.type, MLS.Algorithm))
            graph.add((encoder_uri, RDFS.label, Literal(encoder_name)))
            graph.add((algo_uri, MLSO.hasImageEncoder, encoder_uri))
        elif p_norm == 'text_encoder':
            encoder_name = str(lit)
            encoder_uri = MLFLOW[f'algorithm/{slugify(encoder_name)}']
            graph.add((encoder_uri, RDF.type, MLS.Algorithm))
            graph.add((encoder_uri, RDFS.label, Literal(encoder_name)))
            graph.add((algo_uri, MLSO.hasTextEncoder, encoder_uri))
        elif p_norm == 'embedding_model':
            model_name = str(lit)
            model_uri = MLFLOW[f'algorithm/{slugify(model_name)}']
            graph.add((model_uri, RDF.type, MLS.Algorithm))
            graph.add((model_uri, RDFS.label, Literal(model_name)))
            graph.add((algo_uri, MLSO.hasEmbeddingModel, model_uri))
        else:
            graph.add((algo_uri, prop_uri, lit))
        return

    # 4. Data 级属性
    if p_norm in PARADIGM_PROPERTY_CONFIG['data'] and data_uri:
        prop_name = _get_property_uri(param_key)
        prop_uri = MLSO[prop_name]
        graph.add((data_uri, prop_uri, lit))
        return


def _add_pipeline_relationships(run_uri: URIRef, model_uri: URIRef, algo_uri: URIRef,
                                record: dict, graph: Graph):
    """
    添加 Pipeline 世系关系：
    - LoRA: basedOnModel (adapter → base model)
    - Knowledge Distillation: distilledFrom/distillsTo (student ↔ teacher)
    - Self-Supervised: wasPretrainedFrom, producesRepresentation, usesRepresentation
    """
    tags = record.get('tags', {})
    params = record.get('params', {})
    metrics = record.get('metrics', {})

    # === LoRA: base_model → basedOnModel ===
    base_model = params.get('base_model') or params.get('base-model')
    if base_model:
        base_model_uri = MLFLOW[f'model/{slugify(str(base_model))}']
        graph.add((model_uri, MLSO.basedOnModel, base_model_uri))

    # === Knowledge Distillation: teacher ↔ student ===
    # NOTE: We do NOT depend on algo_uri here. KD relationship is between the
    # actual TEACHER algorithm (e.g. resnet18) and STUDENT algorithm (e.g. mobilenet_v2),
    # NOT the shared "KnowledgeDistillation" paradigm Algorithm.
    teacher_model = params.get('teacher_model') or params.get('teacher-model')
    student_model = params.get('student_model') or params.get('student-model')
    distillation_temp = params.get('distillation_temperature') or params.get('kd_temperature')
    distillation_alpha = params.get('distillation_alpha') or params.get('alpha')
    distillation_loss = params.get('distillation_loss')

    # Determine the "target" algorithm (the one that "owns" the distillation config)
    # Priority: student algorithm if we know it, otherwise teacher algorithm
    if student_model:
        target_algo_uri = MLFLOW[f'algorithm/{slugify(str(student_model))}']
    elif teacher_model:
        target_algo_uri = MLFLOW[f'algorithm/{slugify(str(teacher_model))}']
    else:
        target_algo_uri = None

    if teacher_model:
        # This run is a TEACHER: it teaches student_model
        teacher_algo_uri = MLFLOW[f'algorithm/{slugify(str(teacher_model))}']
        graph.add((teacher_algo_uri, RDF.type, MLS.Algorithm))
        graph.add((teacher_algo_uri, RDFS.label, Literal(str(teacher_model))))

        if student_model:
            # Bidirectional distillation link: teacher ↔ student
            student_algo_uri = MLFLOW[f'algorithm/{slugify(str(student_model))}']
            graph.add((student_algo_uri, RDF.type, MLS.Algorithm))
            graph.add((student_algo_uri, RDFS.label, Literal(str(student_model))))
            graph.add((teacher_algo_uri, MLSO.distillsTo, student_algo_uri))
            graph.add((student_algo_uri, MLSO.distilledFrom, teacher_algo_uri))
            # hasStudentModel on teacher algorithm
            graph.add((teacher_algo_uri, MLSO.hasStudentModel, student_algo_uri))

    elif student_model:
        # This run is a STUDENT: it is distilled FROM teacher_model
        student_algo_uri = MLFLOW[f'algorithm/{slugify(str(student_model))}']
        graph.add((student_algo_uri, RDF.type, MLS.Algorithm))
        graph.add((student_algo_uri, RDFS.label, Literal(str(student_model))))

        teacher_model_name = params.get('teacher_model') or params.get('teacher-model')
        if teacher_model_name:
            teacher_algo_uri = MLFLOW[f'algorithm/{slugify(str(teacher_model_name))}']
            graph.add((teacher_algo_uri, RDF.type, MLS.Algorithm))
            graph.add((teacher_algo_uri, RDFS.label, Literal(str(teacher_model_name))))
            graph.add((student_algo_uri, MLSO.distilledFrom, teacher_algo_uri))
            graph.add((teacher_algo_uri, MLSO.distillsTo, student_algo_uri))
            # hasTeacherModel on student algorithm
            graph.add((student_algo_uri, MLSO.hasTeacherModel, teacher_algo_uri))
        elif algo_uri:
            # Fallback: link via paradigm algo
            graph.add((student_algo_uri, MLSO.distilledFrom, algo_uri))
            graph.add((algo_uri, MLSO.distillsTo, student_algo_uri))

    # Add distillation hyperparameters to the target algorithm
    if target_algo_uri:
        if distillation_temp:
            lit = literal_from_value(distillation_temp)
            if lit is not None:
                graph.add((target_algo_uri, MLSO.hasDistillationTemperature, lit))
        if distillation_alpha:
            lit = literal_from_value(distillation_alpha)
            if lit is not None:
                graph.add((target_algo_uri, MLSO.hasDistillationAlpha, lit))
        if distillation_loss:
            lit = literal_from_value(distillation_loss)
            if lit is not None:
                graph.add((target_algo_uri, MLSO.hasDistillationLoss, lit))


    # === Self-Supervised Learning ===
    learning_stage = tags.get('mlsea.learning_stage')
    if learning_stage:
        if learning_stage.lower() == 'pretext':
            # 前文本任务产生表示
            encoder_name = params.get('encoder_model') or params.get('backbone_model')
            if encoder_name:
                encoder_uri = MLFLOW[f'model/{slugify(str(encoder_name))}']
                graph.add((encoder_uri, RDF.type, MLS.Model))
                graph.add((run_uri, MLSO.producesRepresentation, encoder_uri))

        elif learning_stage.lower() == 'downstream':
            # 下游任务使用表示
            pretext_run_id = tags.get('mlsea.pretrain_run_id')
            if pretext_run_id:
                pretext_uri = MLFLOW[f'run/{pretext_run_id}']
                graph.add((pretext_uri, RDF.type, MLS.Run))
                graph.add((run_uri, MLSO.wasPretrainedFrom, pretext_uri))

            encoder_name = params.get('pretrained_encoder') or params.get('encoder_model')
            if encoder_name:
                encoder_uri = MLFLOW[f'model/{slugify(str(encoder_name))}']
                graph.add((encoder_uri, RDF.type, MLS.Model))
                graph.add((run_uri, MLSO.usesRepresentation, encoder_uri))

    # === Agent: max_iterations → hasMaxIterations ===
    # (已在 _map_paradigm_property 中处理)
