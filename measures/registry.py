from measures.CDR_index import CDR_Index
from measures.CVDD import CVDD
from measures.CVNN import CVNN, CVNN_halkidi
from measures.DCVI import DCV_Index
from measures.DSI import DSI
from measures.ICAV import IC_av
from measures.VIASCKDE import VIASCKDE
from measures.dbcv_measures import DBCV
from measures.standard_measures import Silhouette_Coefficient, VRC, SDBW

registered_measures = [Silhouette_Coefficient, DBCV, DSI, CDR_Index, VIASCKDE, VRC, CVNN_halkidi, SDBW, CVDD, DCV_Index,
                       IC_av, CVNN]


def get_measures():
    return registered_measures


def get_measures_dict():
    return {measure().name: measure() for measure in registered_measures}
