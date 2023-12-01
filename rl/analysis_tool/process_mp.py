import json
import os 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

row_path = "/mnt_data/Dislyte_AIrobot_Data/json_data/v332"
yyy = os.listdir(row_path)
yyy_path = [os.path.join(row_path,x) for x in yyy]

NAMES = ["sSkillType","sResLevel","sAwakenType","sSkillLevel","sOperationType","sTargetCamp","sSkillAIType","bCfgID","bTime","bCfgIsClear","bCfgIsTrans","bCfgNature","oHeloID", "equipSuitData"]

sSkillType = set()
sResLevel = set()
sAwakenType = set()
sSkillLevel = set()
sOperationType = set()
sTargetCamp = set()
sSkillAIType = set()
bCfgID = set()
bTime = set()
bCfgIsClear = set()
bCfgIsTrans = set()
bCfgNature = set()
oHeloID = set()
equipSuitData = set()


for path_o in yyy_path:
    # path_o = "E:\\00XGData\\PVP\\TrainingDataJson\\v326\\2023-07-23"
    path_l = os.listdir(path_o)

    
    def process_file(p):
        with open(os.path.join(path_o, p)) as f:
            try:
                datas = json.load(f)
            except:
                print("========wrong========",os.path.join(path_o,p))
        
                return
        for data in datas:
            for n in ["TeamsCharState","MainCharState","EnemiesCharState"]:
                for k in data[n]:
                    try:
                        oHeloID.add(k["oHeloID"])
                        if k["oHeloID"] == -1 and n == "MainCharState":
                            print(os.path.join(path_o,p),"***********wrong hero id**************")
                        for bf in k["buffData"]:
                            bCfgID.add(bf["bCfgID"])
                            bTime.add(bf["bTime"])
                            bCfgIsClear.add(bf["bCfgIsClear"])
                            bCfgIsTrans.add(bf["bCfgIsTrans"])
                            bCfgNature.add(bf["bCfgNature"])
                        for sf in k["skillData"]:
                            sSkillType.add(sf["sSkillType"])
                            sResLevel.add(sf["sResLevel"])
                            sAwakenType.add(sf["sAwakenType"])
                            sSkillLevel.add(sf["sSkillLevel"])
                            sOperationType.add(sf["sOperationType"])
                            sTargetCamp.add(sf["sTargetCamp"])
                            sSkillAIType.add(sf["sSkillAIType"])
                        for equipid in k["equipSuitData"]:
                            equipSuitData.add(equipid)
                    except:
                        print("========read wrong!========",os.path.join(path_o,p))
            
    with ThreadPoolExecutor(max_workers=4) as executor:  # 使用 4 个线程
        results = list(tqdm(executor.map(process_file, path_l), total=len(path_l)))


print("max sSkillType: ",max(sSkillType),",","min sSkillType: ",min(sSkillType))
print("max sResLevel: ",max(sResLevel),",","min sResLevel: ",min(sResLevel))
print("max sAwakenType: ",max(sAwakenType),",","min sAwakenType: ",min(sAwakenType))
print("max sSkillLevel: ",max(sSkillLevel),",","min sSkillLevel: ",min(sSkillLevel))
print("max sOperationType: ",max(sOperationType),",","min sOperationType: ",min(sOperationType))
print("max sTargetCamp: ",max(sTargetCamp),",","min sTargetCamp: ",min(sTargetCamp))
print("max sSkillAIType: ",max(sSkillAIType),",","min sSkillAIType: ",min(sSkillAIType))
print("list bCfgID: ",list(bCfgID))
print("max bTime: ",max(bTime),",","min bTime: ",min(bTime))
print("max bCfgIsClear: ",max(bCfgIsClear),",","min bCfgIsClear: ",min(bCfgIsClear))
print("max bCfgIsTrans: ",max(bCfgIsTrans),",","min bCfgIsTrans: ",min(bCfgIsTrans))
print("max bCfgNature: ",max(bCfgNature),",","min bCfgNature: ",min(bCfgNature))
print("max oHeloID: ",max(oHeloID),",","min oHeloID: ",min(oHeloID))
print("max equipSuitData", max(equipSuitData),",","min equipSuitData: ",min(equipSuitData))

