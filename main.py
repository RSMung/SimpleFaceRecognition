
from ArcFace.train_arcface import trainArcFaceMain
from ElasticFace.train_elasticface import trainElasticFaceMain
from NormalCls.train_normal_cls import trainNormalClsMain


flag1 = "trainArcFaceMain"
flag2 = "trainNormalClsMain"
flag3 = "trainElasticFaceMain"





# flag = flag1
# flag = flag2
flag = flag3



if flag == flag1:
    trainArcFaceMain()
elif flag == flag2:
    trainNormalClsMain()
elif flag == flag3:
    trainElasticFaceMain()
else:
    raise RuntimeError(f"flag:{flag} is invalid!")