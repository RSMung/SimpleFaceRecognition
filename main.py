
from ArcFace.train_arcface import trainArcFaceMain


flag1 = "trainArcFaceMain"
# flag2 = "testArcFaceMain"
# flag3 = "trainShotMain"





flag = flag1
# flag = flag2
# flag = flag3



if flag == flag1:
    trainArcFaceMain()
# elif flag == flag2:
#     testArcFaceMain()
# elif flag == flag3:
#     trainShotMain()
else:
    raise RuntimeError(f"flag:{flag} is invalid!")