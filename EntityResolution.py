"""
A proposed solution to the Entity Resolution of the 2nd DI2KG Workshop Challenge (Monitor Data, Domain knowledge category)

Authors:
Gabriel Campero Durand (campero@ovgu.de)
Anshu Daur (anshu.daur@gmail.com)
Chinmaya Hegde (chinmaya.hegde@st.ovgu.de)
Prafulla Diwesh (prafulla.diwesh@st.ovgu.de)
Shivalika Suman (shivalika.suman@st.ovgu.de)
Vinayak Kumar (vinayak.kumar@ovgu.de)
David Broneske (david.broneske@ovgu.de)

Affiliation: University of Magdeburg, Databases and Software Engineering Workgroup.
"""
from os import listdir, walk
from os.path import isfile, join
import json
import pickle
import json
import os
import numpy as np
import pandas as pd
import re
import copy
import itertools 

def get_dic_key(val, dic):
    for key, value in dic.items():
        for v in value: 
            if val == v:
                return key.lower() 
    return val.lower()

"""
Step 1: TF-IDF vectorization and collecting of the explicit brands
"""
print("A solution highly tailored to the domain, with the core strategies of: information propagation for model detection, carefully-tuned brand and model extraction (with domain-specific choices). All in all it could take around 15 minutes to run this, depending on the machine.")
print("************************************")
print("Program assumptions (other than requirements):\n 1) We assume 2013_monitor_specs to be in the same path.\n")
print("The domain-specific choices we made are limited to: \na) cleaning of site-specific texts for better TF-IDF results, \nb) non-exhaustive brand (attribute) keywords, extracted by looking at some examples of the data \nc) brand names that were extracted with a bit of a human-in-the-loop process (where we saw the brand names emerging and collected alternative names),\nd) a large amount of rules for brand cleaning, resulting from data understanding (this is the less general aspect of our solution)... the amount of hand-crafted configurations really show the amount of time the team spent exploring and understanding the data, \ne) Rules for extracting the models\nf) Cleaning of false-positive model names.\nWe consider the hard-coded rules to deter from our generality. However, we include them since they are crucial for finding a straight-forward solution with the limited resources chosen.\n\n")


print("Phase 1: Collecting of the explicit brands. (One minute or so)")
dict_={'spec':[],
      'title':[], 
      'brand':[]}
files = sorted([join(r,file).replace("2013_monitor_specs/","").replace(".json","") for r,d,f in walk("2013_monitor_specs/") for file in f])

key_words_brands=['brand','brand name','manufacturer','product name','product line','label','publisher','studio']#We get this by just looking at the data. 
#'aoc envision',
brand_list=['viewSonic','lenovo','dell','hp','lg','formac','iiyama','planar',
           'fujitsu','philips','phillips','nec','acer','westinghouse','allen bradley','asus',
           'samsung','ibm','barco','olympus','panasonic','raritan','elo','acer','apple','eizo','iogear','aoc', 'benq','qnix','mimo','envision','aopen',
            'aten','hannspree','datalux','gnr','veba','hyunday','hyundai','ag neovo','lilliput','princeton',
            'aoc monitor','hanns-g','xenarc','ag neovo','tatung','sharp','PNA601','auria','sva','relisys',
            'doublesight displays','doublesight','lynteck','lenovo thinkvision','hewlett-packard','Hewlett packard', 'hhewlett'
            'proview','genie','wortmann','seacomp','zeiss','intellinet','star','rotronic','neovo','newstar','promethean',
            'eizi','kogan','sunbrite','partnertech','puritron','elitedesktop','hitachi','syncmaster',
           'kontron','liquidvideo','preh','crystalpro','agm','sunbritetv','zentview tomato',
           'timex','bestech','gateaway','triple arm','matrox','hama','tomy classic','contec',
           'kongtop','view sonic','tommee tippee','motorola','tenvis',
           'maxdata','run plus','ikegami','bmw','vector','machood','quanta','rittal',
           'vibox','yuraku','nextbase','ohmeda','daevoo','funtica',
           'performa','cnc','tomy digital','psi','next','skyport',
           'moview','brilliance','sigma','bsi','videoseven','belinea','vaillant',
           'vibox','unho','concept','omnitronic','sgi','emprex','m&a','starlogic','advent',
           'batesias','anyarm','seneye','transcend','unho','baumer','xexun', 'magnavox', 
           'binatone','auna','professional','staples','merit','kortek',
           'yusmart','videoseven','powerworks','medion',
           'mendor','xcellon','tanita','digimate','ryoku','tenvis','likom','sun 365','nit rackmux',
           'westbrook','proac','viglen','austin hughes','slim age','slimage','moview',
           'lexus','urmet arco','advent','peavey eurosys','sandberg','steelmate','stem izon','jetway',
           'rose rackview','atronic','brasscraft','vibrant','aview','skyport','au optronics','matchless',
           'harsper','ilyama','tomy digital','prodisplay','evov','tecklink','skytec','bt pathway',
           'imm living inc.','e yama','vector','contec08d','amstyle','chatsworth','sensormatic','hannsg','v7', 
           'Roline','hanns.g','apc','alpha touch','xo vision','gvision','mcm',"chasis plans",'arbor',"faytech", "chassis plans","chunghwa"] #Domain knowledge and dataset-specific.
brand_list=list(set([_.lower() for _ in brand_list]))
not_brands=['mp3car', 'tft', 'mixed', 'core', 'r&m', 'x-10', 'business','rx', 'skilcraft','nds','pro','vesa','pct2265 22 inch black hid bezel multi-touch lcd usb vga hdmi dvi-d','pxl2430mw','vision','pl1711m', 'acbel','kenwood','krk','l@@k','lindam clarity','lipo','mamba']
rule_dic = {'contec':['contec08d','contec'],'dell':['dell','ultrasharp','optiplex','wyse'],'hp':['hewlett-packard','hewlett packard','hhewlett','zr22w','hp'],'nec':['nec'],
      'mitsubishi':['mitsubishi'],'samsung':['dc series','samsung','essential'],'lg':['lg','sva-usa'],'apple':['apple'],
      'acer':['acer'],'cisco':['cisco'],'philips':['phillips','philips'],'asus':['asus','republic of gamers'],'aver':['aver','avervision'],
       'iiyama':['iiyama'],'epson':['epson'],'compaq':['compaq'],'sony':['sony'],'fuji':['fuji','fujicom'],
       'viewsonic':['viewsonic','optiquest'],'hyundai':['hyundai','hyunday'],'lenovo':['lenovo','d221','thinkpad'],'ibm':['ibm'],
      'elo':['elo','tyco electronics'],'fujitsu':['fujitsu','fijitsu'],'packard bell':['packard bell'],'Yiynova':['panda city'],
       'datalogic':['datalogic'],'3m':['microtouch','3m'],'touchsystems':['touch systems','touchsystems'],'panasonic':['panasonic'],'olympus':['olympus'],
      'american dynamics':['american dynamics'],'ctx':['ctx'],'pioneerpos':['pioneerpos'],'ais':['ais'],'adesso':['adesso'],
      'advance one':['advance one'],'ace':['ace'],'aei':['aei'],'ag neovo':['ag neovo technology corp.'],'APC':['american power conversion corp','apc'],
      'aspen':['aspen'],'aten technologies':['aten technologies'],'automation direct':['automation direct'],'bci technology':['bci technology'],
      'cleartunes':['cleartunes'],'cinq':['cinq'],'cornea':['cornea'],'element electronics':['element electronics'],'ematic':['ematic'],
      'emerson':['emerson'],'etronix':['etronix'],'feelworld':['feelworld'],'foxboro':['foxboro'],'innolux':['innolux'],
      'infocus':['infocus'],'logic controls':['logic controls'],'magnavox':['magnavox'],'norwood':['norwood'],'nanovision':['nanovision'],
      'omron':['omron'],'pelco':['pelco'],'insignia':['insignia'],'crossover':['crossover'],'mag':['mag innovision','mag'],'bk sems':['bk sems'],
      'avocent':['avocent'],'sceptre':['sceptre'],'chimei':['chimei'],'kds':['kds'],'siemens':['siemens'],'totevision':['totevision'],
      'envision monitors':['envision monitors'],'x-star':['x-star'],'sympodium':['sympodium'],'aoc':['brand new b-billion','aoc international','aoc monitor','aoc envision'],'apc':['apc'],
      'i inc':['i inc'],'amptron':['amptron'],'gechic':['gechic'],'amw':['amw'],'pyle':['Pyle audio','inc'],'adi':['adi'],
      'cmo':['cmo'],'vaddio':['vaddio'],'upstar':['upstar'],'commodore':['commodore'],'sansui':['sansui'],'carnetix':['carnetix'],
      'earth computer tech':['earth computer tech'],'itronix':['itronix'],'planar systems':['planar systems'],'schneider electric':['schneider electric'],
      'vizta':['vizta'],'achieva':['achieva'],'avue':['avue'],'soyo':['soyo'],'emachines':['emachines'],'gvision':['gvision'],
      'silicon graphics':['sgi','silicon graphics'],'starlogic':['star logic'],'mimo':['mimo monitors'],'hannspree':['hanns-g','hanns.g','hannspree'],
      'hkc':['hkc'],'sansui':['sansui'],'omni vision':['omni vision'],'proview':['proview system desktop'],'pyle':['pylehome'],
      'evga':['evga'],'formac':['formac'],'planar':['helium'],'eizo':['eizo coloredge','eizo'],'edge10':['edge10'],"chassis plans":['chasis plans','chassis plans'],"medion":['medion erazer','medion']}

brand_dict=dict()
count=0

words_re = re.compile("|".join(brand_list))
for item in files:
    f = open("2013_monitor_specs/"+item+".json")
    data = json.load(f)
    found_brand=False
    title=""
    for i in sorted(list(data.keys())):
        str_to_append=str(data[i])
        if i=="<page title>":
            if str_to_append[0:4]=="Buy ":
                str_to_append=str_to_append[4:]
            str_to_append=str_to_append.replace("PCPartPicker Canada","")
            str_to_append=str_to_append.replace("Pricing and Details - Catalog.com","")
            str_to_append=str_to_append.replace("Yikus.com","")
            str_to_append=str_to_append.replace("| UK | Ebay Items | Best Deal Items | Auctions | Free Shipping | Goods | Services | Wholesales | Retail | Trade","")
            str_to_append=str_to_append.replace("| eBay","")
            str_to_append=str_to_append.replace("| Compare Prices & Save shopping in Australia","")
            str_to_append=str_to_append.replace("- MediaShopUK","")
            str_to_append=str_to_append.replace("- MrHighTech Shop","")
            str_to_append=str_to_append.replace(": Monitors : Monitors : Nexus Technology UK, Colchester, Essex","")
            str_to_append=str_to_append.replace("by Office Depot","")
            str_to_append=str_to_append.replace("OHC24 Shop : Monitor > Monitor search help >","")
            str_to_append=str_to_append.replace("- PC-Canada.com","")
            str_to_append=str_to_append.replace("- SoftwareCity.ca - Canada","")
            str_to_append=str_to_append.replace("- Vology","")
            str_to_append=str_to_append.replace("- Xpcpro.com","")
            str_to_append=str_to_append.replace("today at PC Connection","")
            str_to_append=str_to_append.replace("Compare prices for","")
            str_to_append=str_to_append.replace("reviews - ShopMania","")
            title=str_to_append
        if i in key_words_brands and not found_brand:
            found_brand=True
            v=data[i]
            if type(v)==list:
                v=v[0].lower()
            v=v.lower()
            if ':' in v:
                v=v.replace(':','').strip()
            brand=get_dic_key(v, rule_dic)
            if 'acer' in brand:
                brand='acer'
            elif '3m' in brand:
                brand='3m'
            elif 'benq' in brand:
                brand='benq'
            elif 'asus' in brand:
                brand='asus'
            elif 'barco' in brand:
                brand='barco'
            elif 'dell' in brand:
                brand='dell'
            elif 'doublesight' in brand:
                brand='doublesight'
            elif 'elo' in brand:
                brand='elo'
            elif 'envision' in brand and not brand=="aoc envision":
                brand='envision'
            elif "aoc" in brand:
                brand="aoc"
            elif 'hewlett' in brand or 'hp' in brand:
                brand='hp'
            elif 'hyundai' in brand:
                brand='hyundai'
            elif 'iiyama' in brand:
                brand='iiyama'
            elif 'lenovo' in brand or 'thinkpad' in brand or 'think pad' in brand:
                brand='lenovo'
            elif 'lg' in brand:
                brand='lg'
            elif 'nec' in brand:
                brand='nec'
            elif 'philips' in brand:
                brand='philips'
            elif 'planar' in brand:
                brand='planar'
            elif 'princeton' in brand:
                brand='princeton'
            elif 'pyle' in brand:
                brand='pyle'
            elif 'rockwell' in brand:
                brand='rockwell'
            elif 'viewsonic' in brand:
                brand='viewsonic'
            elif brand in not_brands:
                brand='to_delete'
            dict_['brand'].append(brand)
            brand_dict[item]=brand
            if brand not in brand_list:
                brand_list.append(brand)
                words_re = re.compile("|".join(brand_list))
    dict_['spec'].append(item)
    dict_['title'].append(title)
    if not found_brand:
        dict_['brand'].append("N.A")
        brand_dict[item]="N.A"

for i in range(0,len(dict_["spec"])):
   if dict_['brand'][i]== "N.A":
       title=dict_['title'][i].lower()
       if words_re.search(title):
           brand=words_re.search(title).group()
           brand=get_dic_key(brand, rule_dic)
           if 'acer' in brand:
               brand='acer'
           elif '3m' in brand:
               brand='3m'
           elif 'asus' in brand:
               brand='asus'
           elif 'benq' in brand:
               brand='benq'
           elif 'barco' in brand:
               brand='barco'
           elif 'dell' in brand:
               brand='dell'
           elif 'doublesight' in brand:
               brand='doublesight'
           elif 'elo' in brand: 
               brand='elo'
           elif 'envision' in brand and not brand=="aoc envision":
               brand='envision'
           elif "aoc" in brand:
               brand="aoc"
           elif 'hewlett' in brand or 'hp' in brand:
               brand='hp'
           elif 'hyundai' in brand:
               brand='hyundai'
           elif 'iiyama' in brand:
               brand='iiyama'
           elif 'lenovo' in brand or 'thinkpad' in brand or 'think pad' in brand:
               brand='lenovo'
           elif 'lg' in brand:
               brand='lg'
           elif 'nec' in brand:
               brand='nec'
           elif 'philips' in brand:
               brand='philips'
           elif 'planar' in brand:
               brand='planar'
           elif 'princeton' in brand:
               brand='princeton'
           elif 'pyle' in brand:
               brand='pyle'
           elif 'rockwell' in brand:
               brand='rockwell'
           elif 'viewsonic' in brand.lower():
               brand='viewsonic'
           elif brand in not_brands:
               brand='to_delete'
           dict_['brand'][i]=brand
           brand_dict[dict_["spec"][i]]=brand

"""
Step 2: Brand and non-product cleaning
"""
print("Phase 2: Brand cleaning- In this step we adopt a series of rules to clean brands and non-products. (One minute or so)")

to_delete_rules=set(['belkin','iogear'])
to_delete=set(['ergotron','compucessory','fellowes','kantek','rosewill','sports tracker',"dust off", "dust_off","dust-off"])
to_delete_files=set()
to_delete_files_rules=set()
reasons_to_delete=dict()
item_to_id = dict()
stopwords=["17","15","display","product","led","17'","15'","21","23","24","3m","22","17","21.5","22","ve","20","allen-bradley","cd/mâ²", "color","professional","details","ultrasharp","27","3d","cinema","wide","x series","ultrasharp","19","920","ds-","new","cd/m2","1u","ohc24","2711p", "715l1009", "pe1229", "17''","e114849"]

for i in brand_dict:
    if brand_dict[i] in to_delete:
        to_delete_files.add(i)    
    elif brand_dict[i] in to_delete_rules:
        to_delete_files_rules.add(i)

to_delete_files.add("www.ebay.com/19097")

brand_dict2= {
   "pge":["pge"],
   "gvision":["gvision"],
   "chassis plans": ["chassis plans", "chasis plans"],
    "auo":["auo","au optronics"],
   "mcm":["MCM","mcm"],
   "dell":[
      "dell",
      "ultrasharp",
      "optiplex",
      "wyse"
   ],
   "hp":[
      "hewlett-packard",
      "hewlett packard",
      "hhewlett",
      "zr22w"
   ],
   "nec":[
      "nec"
   ],
   "mitsubishi":[
      "mitsubishi"
   ],
   "samsung":[
      "dc series",
      "samsung",
      "essential"
   ],
   "lg":[
      "lg",
      "sva-usa"
   ],
   "apple":[
      "apple",
      "macbook"
   ],
   "acer":[
      "acer"
   ],
   "cisco":[
      "cisco"
   ],
   "philips":[
      "phillips",
      "philips"
   ],
   "asus":[
      "asus",
      "republic of gamers"
   ],
   "aver":[
      "aver",
      "avervision"
   ],
   "iiyama":[
      "iiyama",
      "ilyama"
   ],
   "epson":[
      "epson"
   ],
   "compaq":[
      "compaq"
   ],
   "sony":[
      "sony"
   ],
   "fuji":[
      "fuji",
      "fujicom"
   ],
   "viewsonic":[
      "viewsonic",
      "optiquest"
   ],
   "hyundai":[
      "hyundai",
      "hyunday"
   ],
   "lenovo":[
      "lenovo",
      "d221",
      "thinkpad"
   ],
   "ibm":[
      "ibm"
   ],
   "elo":[
      "elo",
      "tyco electronics"
   ],
   "fujitsu":[
      "fujitsu",
      "fijitsu"
   ],
   "packard bell":[
      "packard bell"
   ],
   "Yiynova":[
      "panda city"
   ],
   "datalogic":[
      "datalogic"
   ],
   "3M":[
      "microtouch"
   ],
   "touchsystems":[
      "touch systems",
      "touchsystems"
   ],
   "panasonic":[
      "panasonic"
   ],
   "olympus":[
      "olympus"
   ],
   "american dynamics":[
      "american dynamics"
   ],
   "ctx":[
      "ctx"
   ],
   "pioneerpos":[
      "pioneerpos"
   ],
   "ais":[
      "ais"
   ],
   "adesso":[
      "adesso"
   ],
   "advance one":[
      "advance one"
   ],
   "ace":[
      "ace"
   ],
   "aei":[
      "aei"
   ],
   "ag neovo":[
      "ag neovo technology corp."
   ],
   "apc":[
      "american power conversion corp"
   ],
   "aspen":[
      "aspen"
   ],
   "aten technologies":[
      "aten technologies",
      "aten corp",
      "aten"
   ],
   "automation direct":[
      "automation direct"
   ],
   "bci technology":[
      "bci technology"
   ],
   "cleartunes":[
      "cleartunes"
   ],
   "cinq":[
      "cinq"
   ],
   "cornea":[
      "cornea"
   ],
   "element electronics":[
      "element electronics"
   ],
   "ematic":[
      "ematic"
   ],
   "emerson":[
      "emerson"
   ],
   "etronix":[
      "etronix"
   ],
   "feelworld":[
      "feelworld"
   ],
   "foxboro":[
      "foxboro"
   ],
   "innolux":[
      "innolux"
   ],
   "infocus":[
      "infocus"
   ],
   "logic controls":[
      "logic controls"
   ],
   "magnavox":[
      "magnavox"
   ],
   "norwood":[
      "norwood"
   ],
   "nanovision":[
      "nanovision",
      "nano"
   ],
   "omron":[
      "omron"
   ],
   "pelco":[
      "pelco"
   ],
   "insignia":[
      "insignia"
   ],
   "crossover":[
      "crossover"
   ],
   "mag":[
      "mag innovision",
      "mag"
   ],
   "bk sems":[
      "bk sems"
   ],
   "avocent":[
      "avocent"
   ],
   "sceptre":[
      "sceptre"
   ],
   "chimei":[
      "chimei"
   ],
   "kds":[
      "kds"
   ],
   "siemens":[
      "siemens"
   ],
   "totevision":[
      "totevision",
      "TOTE VISION",
      "tote vision"
   ],
   "envision monitors":[
      "envision monitors"
   ],
   "x-star":[
      "x-star"
   ],
   "sympodium":[
      "sympodium"
   ],
   "aoc":[
      "brand new b-billion",
      "aoc international",
      "apc",
      "aoc monitor",
      "aoc envision"
   ],
   "i inc":[
      "i inc"
   ],
   "amptron":[
      "amptron"
   ],
   "gechic":[
      "gechic"
   ],
   "amw":[
      "amw"
   ],
   "pyle":[
      "Pyle audio",
      "inc"
   ],
   "adi":[
      "adi"
   ],
   "cmo":[
      "cmo"
   ],
   "vaddio":[
      "vaddio"
   ],
   "upstar":[
      "upstar"
   ],
   "commodore":[
      "commodore"
   ],
   "sansui":[
      "sansui"
   ],
   "carnetix":[
      "carnetix"
   ],
   "earth computer tech":[
      "earth computer tech"
   ],
   "itronix":[
      "itronix"
   ],
   "planar systems":[
      "planar systems"
   ],
   "schneider electric":[
      "schneider electric"
   ],
   "vizta":[
      "vizta"
   ],
   "achieva":[
      "achieva"
   ],
   "avue":[
      "avue"
   ],
   "soyo":[
      "soyo"
   ],
   "emachines":[
      "emachines"
   ],
   "gvision":[
      "gvision"
   ],
   "hkc":[
      "hkc"
   ],
   "sansui":[
      "sansui"
   ],
   "omni vision":[
      "omni vision"
   ],
   "proview":[
      "proview system desktop"
   ],
   "pyle":[
      "pylehome"
   ],
   "silicon graphics":[
      "sgi",
      "silicon graphics"
   ],
   "starlogic":[
      "star logic"
   ],
   "mimo":[
      "mimo monitors"
   ],
   "hannspree":[
      "hanns-g",
      "hanns.g",
      "hannspree",
      "hannsg",
      "hanns g",
   ],
   "evga":[
      "evga"
   ],
   "formac":[
      "formac"
   ],
   "planar":[
      "helium"
   ],
   "eizo":[
      "eizo coloredge",
      "eizo"
   ],
   "edge10":[
      "edge10"
   ],
   "xo vision":["xo vision"],
   "faytech": ["faytech"],
   "tatung":["tatung"],
   "primera":["primera"],
"polycom":["polycom"],
"yuraku":["yuraku"],
}

quick_fix={"aoc":"aoc",
"hanns g":"hannspree",
"alienware":"alienware",
"barco":"barco",
"ultrasharp u2410":"dell",
"toshiba":"toshiba","xerox":"xerox","compaq v":"compaq","benq":"benq", 
"unbranded/generic":"unbranded/generic", "ncr":"ncr", "yamakasi":"yamakasi", "gateway":"gateway", "rockwell":"rockwell", "allen bradley":"rockwell","qnix":"qnix", "night owl":"night owl", "1plus":"1plus","wasabi mango":"wasabi mango","speco":"speco","bosto":"bosto","nanov":"nanov","westinghouse":"westinghouse","veba":"veba","pioneer pos":"pioneerpos","3m":"3m", "samsung syncmaster":"samsung", "bk sems by samsung":"samsung","autonav":"autonav","v7":"v7","tatung":"tatung","gnr":"gnr","doublesight":"doublesight","smart technologies":"smart technologies","mace":"mace","prism":"prism","lilliput":"lilliput","monoprice":"monoprice","genie":"genie","i-inc":"i-inc","ctl":"ctl","datalux":"datalux",
"auria":"auria","weldex":"weldex","innovera":"innovera","generaltouch":"generaltouch","atlona":"atlona","aten":"aten technologies","miracle business":"miracle business", "sysonic/miracle business":"miracle business","sysonic":"miracle business", "relisys":"relisys","jvc":"jvc","vigilant":"vigilant","canvys":"canvys","vizio":"vizio","pos-x":"pos-x","microtek":"microtek","viewz":"viewz","xenarc":"xenarc",
"lacie":"lacie","newline interactive":"newline interactive","okina":"okina","raritan":"raritan","princeton":"princeton","dclcd":"dclcd","sgi / silicon graphics":"silicon graphics","rog":"asus","aopen":"aopen","lyntek":"lyntek","firebox":"firebox","startech":"startech","startech.com":"startech", "StarTech":"startech", "marshall":"marshall","idesign":"idesign", "tripp":"tripp","boe hydis":"boehydis", "boehydis":"boehydis","medion":"medion","angel":"angel","advueu":"advueu","ingram":"ingram","norcent":"norcent","wren":"wren","xeno":"xeno", "sun microsystems":"sun microsystems","sunray":"sunray","emprex":"emprex"}
missing_brands=set()
items_with_missing_brands=set()
for item in brand_dict.keys():
    if not brand_dict[item] in brand_dict2:
        if brand_dict[item] in quick_fix:
            brand_dict[item] = quick_fix[brand_dict[item]]
        else:
            if (not brand_dict[item] in to_delete) and (not brand_dict[item] in to_delete_rules) and brand_dict[item]!="to_delete":
                missing_brands.add(brand_dict[item])
                items_with_missing_brands.add(item)

will_delete=[]
will_keep=[]
will_keep_brands=[]

will_keep=['catalog.com/266', 'catalog.com/337', 'catalog.com/402', 'catalog.com/619', 'ce.yikus.com/181', 'ce.yikus.com/537', 'www.best-deal-items.com/1005', 'www.best-deal-items.com/1011', 'www.best-deal-items.com/1033', 'www.best-deal-items.com/1041', 'www.best-deal-items.com/1071', 'www.best-deal-items.com/1085', 'www.best-deal-items.com/1123', 'www.best-deal-items.com/1214', 'www.best-deal-items.com/1229', 'www.best-deal-items.com/1286', 'www.best-deal-items.com/1287', 'www.best-deal-items.com/1324', 'www.best-deal-items.com/1335', 'www.best-deal-items.com/1352', 'www.best-deal-items.com/138', 'www.best-deal-items.com/1382', 'www.best-deal-items.com/1394', 'www.best-deal-items.com/1480', 'www.best-deal-items.com/1517', 'www.best-deal-items.com/1558', 'www.best-deal-items.com/1573', 'www.best-deal-items.com/1588', 'www.best-deal-items.com/159', 'www.best-deal-items.com/1679', 'www.best-deal-items.com/1717', 'www.best-deal-items.com/1788', 'www.best-deal-items.com/18', 'www.best-deal-items.com/1830', 'www.best-deal-items.com/1842', 'www.best-deal-items.com/1844', 'www.best-deal-items.com/1964', 'www.best-deal-items.com/2037', 'www.best-deal-items.com/2038', 'www.best-deal-items.com/2057', 'www.best-deal-items.com/2105', 'www.best-deal-items.com/2120', 'www.best-deal-items.com/2157', 'www.best-deal-items.com/2218', 'www.best-deal-items.com/227', 'www.best-deal-items.com/2365', 'www.best-deal-items.com/239', 'www.best-deal-items.com/2537', 'www.best-deal-items.com/2593', 'www.best-deal-items.com/2657', 'www.best-deal-items.com/2691', 'www.best-deal-items.com/2719', 'www.best-deal-items.com/2777', 'www.best-deal-items.com/28', 'www.best-deal-items.com/282', 'www.best-deal-items.com/430', 'www.best-deal-items.com/437', 'www.best-deal-items.com/509', 'www.best-deal-items.com/526', 'www.best-deal-items.com/591', 'www.best-deal-items.com/598', 'www.best-deal-items.com/628', 'www.best-deal-items.com/658', 'www.best-deal-items.com/702', 'www.best-deal-items.com/73', 'www.best-deal-items.com/742', 'www.best-deal-items.com/773', 'www.best-deal-items.com/780', 'www.best-deal-items.com/783', 'www.best-deal-items.com/857', 'www.best-deal-items.com/873', 'www.best-deal-items.com/969', 'www.cleverboxes.com/221', 'www.cleverboxes.com/516', 'www.ebay.com/10733', 'www.ebay.com/11317', 'www.ebay.com/11500', 'www.ebay.com/11630', 'www.ebay.com/15548', 'www.ebay.com/15955', 'www.ebay.com/16419', 'www.ebay.com/16801', 'www.ebay.com/18221', 'www.ebay.com/18538', 'www.ebay.com/18821', 'www.ebay.com/18895', 'www.ebay.com/18994', 'www.ebay.com/19324', 'www.ebay.com/19325', 'www.ebay.com/19400', 'www.ebay.com/19615', 'www.ebay.com/20048', 'www.ebay.com/20059', 'www.ebay.com/20238', 'www.ebay.com/20521', 'www.ebay.com/20534', 'www.ebay.com/20750', 'www.ebay.com/20913', 'www.ebay.com/21303', 'www.ebay.com/21325', 'www.ebay.com/21402', 'www.ebay.com/21445', 'www.ebay.com/21560', 'www.ebay.com/21566', 'www.ebay.com/21913', 'www.ebay.com/21991', 'www.ebay.com/22344', 'www.ebay.com/22569', 'www.ebay.com/22696', 'www.ebay.com/23445', 'www.ebay.com/23677', 'www.ebay.com/9507', 'www.ebay.com/9572', 'www.ebay.com/9639', 'www.ebay.com/9800', 'www.ebay.com/9809', 'www.pc-canada.com/315', 'www.pcconnection.com/1901', 'www.softwarecity.ca/1533']
will_keep_brands=['startech', 'startech', 'startech', 'startech', 'angel', 'angel', 'unbranded/generic', 'startech', 'acer', 'tripp', 'unbranded/generic', 'starlogic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'boehydis', 'medion', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'advueu', 'boscam', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'ingram', 'startech', 'unbranded/generic', 'norcent', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'tripp', 'unbranded/generic', 'unbranded/generic', 'nec', 'unbranded/generic', 'wren', 'unbranded/generic', 'xeno', 'sun microsystems', 'sunray', 'dell', 'unbranded/generic', 'tripp', 'unbranded/generic', 'acer', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'tripp', 'startech', 'startech', 'emprex', 'unbranded/generic', 'startech', 'unbranded/generic', 'unbranded/generic', 'startech', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'dell', 'ag neovo', 'ag neovo', 'touch controls inc', 'unbranded/generic', 'rca', 'mass multimedia', 'startech', 'arbor', 'touch controls inc', 'lenovo', 'touch displays', 'startech', 'dell', 'suntomo', 'tripp', 'wacom', 'unbranded/generic', 'norcent', 'startech', 'startech', 'dell', 'short-circuit.com', 'tripp', 'tripp', 'niko', 'startech', 'startech', 'tripp', 'unbranded/generic', 'angel', 'lilliput', 'tripp', 'ic power', 'tripp', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'tripp', 'hp', 'unbranded/generic', 'tripp', 'mgc', 'advueu', 'hp', 'tripp', 'hp']

for i in range(0,len(will_keep)):
    brand_dict[will_keep[i]]=will_keep_brands[i]

for item in to_delete_files:
    del brand_dict[item]

for item in to_delete_files_rules:
   with open("2013_monitor_specs/"+item+".json", 'r') as myfile:
       data = myfile.read().lower()
       if not "rack" in data:
           del brand_dict[item]

will_keep2=['catalog.com/323', 'catalog.com/397', 'ce.yikus.com/264', 'ce.yikus.com/303', 'ce.yikus.com/814', 'www.best-deal-items.com/1023', 'www.best-deal-items.com/1037', 'www.best-deal-items.com/107', 'www.best-deal-items.com/1078', 'www.best-deal-items.com/1092', 'www.best-deal-items.com/1115', 'www.best-deal-items.com/1127', 'www.best-deal-items.com/1163', 'www.best-deal-items.com/1166', 'www.best-deal-items.com/1169', 'www.best-deal-items.com/1185', 'www.best-deal-items.com/1188', 'www.best-deal-items.com/1192', 'www.best-deal-items.com/1209', 'www.best-deal-items.com/1225', 'www.best-deal-items.com/1226', 'www.best-deal-items.com/1230', 'www.best-deal-items.com/1277', 'www.best-deal-items.com/1282', 'www.best-deal-items.com/1298', 'www.best-deal-items.com/1319', 'www.best-deal-items.com/1329', 'www.best-deal-items.com/1331', 'www.best-deal-items.com/1343', 'www.best-deal-items.com/1365', 'www.best-deal-items.com/1373', 'www.best-deal-items.com/1425', 'www.best-deal-items.com/1443', 'www.best-deal-items.com/1444', 'www.best-deal-items.com/1454', 'www.best-deal-items.com/1481', 'www.best-deal-items.com/1501', 'www.best-deal-items.com/1503', 'www.best-deal-items.com/1519', 'www.best-deal-items.com/1527', 'www.best-deal-items.com/155', 'www.best-deal-items.com/1559', 'www.best-deal-items.com/1602', 'www.best-deal-items.com/1620', 'www.best-deal-items.com/1625', 'www.best-deal-items.com/1626', 'www.best-deal-items.com/1630', 'www.best-deal-items.com/1645', 'www.best-deal-items.com/1656', 'www.best-deal-items.com/166', 'www.best-deal-items.com/1696', 'www.best-deal-items.com/1701', 'www.best-deal-items.com/1715', 'www.best-deal-items.com/1728', 'www.best-deal-items.com/1744', 'www.best-deal-items.com/1757', 'www.best-deal-items.com/1758', 'www.best-deal-items.com/1769', 'www.best-deal-items.com/1770', 'www.best-deal-items.com/1782', 'www.best-deal-items.com/179', 'www.best-deal-items.com/1821', 'www.best-deal-items.com/1823', 'www.best-deal-items.com/1824', 'www.best-deal-items.com/1826', 'www.best-deal-items.com/1848', 'www.best-deal-items.com/1854', 'www.best-deal-items.com/188', 'www.best-deal-items.com/1881', 'www.best-deal-items.com/1888', 'www.best-deal-items.com/189', 'www.best-deal-items.com/1906', 'www.best-deal-items.com/1929', 'www.best-deal-items.com/1940', 'www.best-deal-items.com/1942', 'www.best-deal-items.com/1953', 'www.best-deal-items.com/1957', 'www.best-deal-items.com/1972', 'www.best-deal-items.com/1988', 'www.best-deal-items.com/2003', 'www.best-deal-items.com/2013', 'www.best-deal-items.com/2021', 'www.best-deal-items.com/2060', 'www.best-deal-items.com/2062', 'www.best-deal-items.com/2071', 'www.best-deal-items.com/2095', 'www.best-deal-items.com/2098', 'www.best-deal-items.com/2132', 'www.best-deal-items.com/2133', 'www.best-deal-items.com/2138', 'www.best-deal-items.com/2139', 'www.best-deal-items.com/2151', 'www.best-deal-items.com/2170', 'www.best-deal-items.com/2190', 'www.best-deal-items.com/2199', 'www.best-deal-items.com/2213', 'www.best-deal-items.com/2220', 'www.best-deal-items.com/224', 'www.best-deal-items.com/2260', 'www.best-deal-items.com/2271', 'www.best-deal-items.com/2290', 'www.best-deal-items.com/2294', 'www.best-deal-items.com/2304', 'www.best-deal-items.com/231', 'www.best-deal-items.com/2325', 'www.best-deal-items.com/2340', 'www.best-deal-items.com/2344', 'www.best-deal-items.com/2348', 'www.best-deal-items.com/2364', 'www.best-deal-items.com/2413', 'www.best-deal-items.com/2423', 'www.best-deal-items.com/2424', 'www.best-deal-items.com/2441', 'www.best-deal-items.com/2459', 'www.best-deal-items.com/2469', 'www.best-deal-items.com/2471', 'www.best-deal-items.com/2504', 'www.best-deal-items.com/2509', 'www.best-deal-items.com/2530', 'www.best-deal-items.com/2534', 'www.best-deal-items.com/2570', 'www.best-deal-items.com/2577', 'www.best-deal-items.com/2578', 'www.best-deal-items.com/2607', 'www.best-deal-items.com/2614', 'www.best-deal-items.com/2616', 'www.best-deal-items.com/2661', 'www.best-deal-items.com/2662', 'www.best-deal-items.com/2669', 'www.best-deal-items.com/267', 'www.best-deal-items.com/2674', 'www.best-deal-items.com/2685', 'www.best-deal-items.com/2690', 'www.best-deal-items.com/2700', 'www.best-deal-items.com/2718', 'www.best-deal-items.com/274', 'www.best-deal-items.com/2768', 'www.best-deal-items.com/2776', 'www.best-deal-items.com/286', 'www.best-deal-items.com/293', 'www.best-deal-items.com/305', 'www.best-deal-items.com/385', 'www.best-deal-items.com/386', 'www.best-deal-items.com/396', 'www.best-deal-items.com/412', 'www.best-deal-items.com/424', 'www.best-deal-items.com/429', 'www.best-deal-items.com/520', 'www.best-deal-items.com/536', 'www.best-deal-items.com/546', 'www.best-deal-items.com/562', 'www.best-deal-items.com/580', 'www.best-deal-items.com/59', 'www.best-deal-items.com/592', 'www.best-deal-items.com/624', 'www.best-deal-items.com/634', 'www.best-deal-items.com/644', 'www.best-deal-items.com/666', 'www.best-deal-items.com/668', 'www.best-deal-items.com/69', 'www.best-deal-items.com/691', 'www.best-deal-items.com/701', 'www.best-deal-items.com/728', 'www.best-deal-items.com/729', 'www.best-deal-items.com/753', 'www.best-deal-items.com/761', 'www.best-deal-items.com/772', 'www.best-deal-items.com/787', 'www.best-deal-items.com/795', 'www.best-deal-items.com/811', 'www.best-deal-items.com/821', 'www.best-deal-items.com/822', 'www.best-deal-items.com/843', 'www.best-deal-items.com/854', 'www.best-deal-items.com/858', 'www.best-deal-items.com/870', 'www.best-deal-items.com/871', 'www.best-deal-items.com/874', 'www.best-deal-items.com/882', 'www.best-deal-items.com/891', 'www.best-deal-items.com/899', 'www.best-deal-items.com/906', 'www.best-deal-items.com/912', 'www.best-deal-items.com/917', 'www.best-deal-items.com/918', 'www.best-deal-items.com/948', 'www.best-deal-items.com/950', 'www.best-deal-items.com/978', 'www.best-deal-items.com/998', 'www.cleverboxes.com/206', 'www.cleverboxes.com/224', 'www.cleverboxes.com/291', 'www.cleverboxes.com/365', 'www.cleverboxes.com/368', 'www.cleverboxes.com/402', 'www.cleverboxes.com/481', 'www.cleverboxes.com/494', 'www.cleverboxes.com/63', 'www.cleverboxes.com/76', 'www.ebay.com/11016', 'www.ebay.com/11081', 'www.ebay.com/11203', 'www.ebay.com/11358', 'www.ebay.com/11443', 'www.ebay.com/11607', 'www.ebay.com/11794', 'www.ebay.com/11964', 'www.ebay.com/12114', 'www.ebay.com/14346', 'www.ebay.com/14551', 'www.ebay.com/14637', 'www.ebay.com/14697', 'www.ebay.com/15377', 'www.ebay.com/15481', 'www.ebay.com/15615', 'www.ebay.com/15648', 'www.ebay.com/16929', 'www.ebay.com/17016', 'www.ebay.com/17040', 'www.ebay.com/18090', 'www.ebay.com/18222', 'www.ebay.com/18303', 'www.ebay.com/18351', 'www.ebay.com/18797', 'www.ebay.com/18833', 'www.ebay.com/19014', 'www.ebay.com/19015', 'www.ebay.com/19097', 'www.ebay.com/19450', 'www.ebay.com/19527', 'www.ebay.com/19944', 'www.ebay.com/19961', 'www.ebay.com/20097', 'www.ebay.com/20159', 'www.ebay.com/20232', 'www.ebay.com/20416', 'www.ebay.com/20454', 'www.ebay.com/20565', 'www.ebay.com/20616', 'www.ebay.com/20686', 'www.ebay.com/20687', 'www.ebay.com/20767', 'www.ebay.com/21071', 'www.ebay.com/21078', 'www.ebay.com/21117', 'www.ebay.com/21298', 'www.ebay.com/21621', 'www.ebay.com/21634', 'www.ebay.com/21637', 'www.ebay.com/21666', 'www.ebay.com/21750', 'www.ebay.com/21893', 'www.ebay.com/21917', 'www.ebay.com/22297', 'www.ebay.com/22677', 'www.ebay.com/22811', 'www.ebay.com/22895', 'www.ebay.com/23034', 'www.ebay.com/23153', 'www.ebay.com/23179', 'www.ebay.com/23182', 'www.ebay.com/23300', 'www.ebay.com/23489', 'www.ebay.com/23684', 'www.ebay.com/9352', 'www.ebay.com/9914', 'www.getprice.com.au/148', 'www.getprice.com.au/195', 'www.getprice.com.au/226', 'www.getprice.com.au/237', 'www.getprice.com.au/239', 'www.getprice.com.au/286', 'www.getprice.com.au/298', 'www.getprice.com.au/304', 'www.getprice.com.au/306', 'www.getprice.com.au/332', 'www.getprice.com.au/336', 'www.getprice.com.au/347', 'www.getprice.com.au/376', 'www.hardware-planet.it/110', 'www.hardware-planet.it/111', 'www.hardware-planet.it/245', 'www.hardware-planet.it/5', 'www.hardware-planet.it/7', 'www.jrlinton.co.uk/1049', 'www.jrlinton.co.uk/1727', 'www.jrlinton.co.uk/464', 'www.jrlinton.co.uk/472', 'www.jrlinton.co.uk/632', 'www.jrlinton.co.uk/778', 'www.jrlinton.co.uk/817', 'www.jrlinton.co.uk/828', 'www.jrlinton.co.uk/836', 'www.jrlinton.co.uk/846', 'www.jrlinton.co.uk/860', 'www.jrlinton.co.uk/862', 'www.jrlinton.co.uk/888', 'www.jrlinton.co.uk/966', 'www.kingsfieldcomputers.co.uk/149', 'www.kingsfieldcomputers.co.uk/197', 'www.kingsfieldcomputers.co.uk/277', 'www.kingsfieldcomputers.co.uk/328', 'www.kingsfieldcomputers.co.uk/652', 'www.kingsfieldcomputers.co.uk/98', 'www.mediashopuk.com/0', 'www.mediashopuk.com/101', 'www.mediashopuk.com/104', 'www.mediashopuk.com/108', 'www.mediashopuk.com/112', 'www.mediashopuk.com/123', 'www.mediashopuk.com/13', 'www.mediashopuk.com/133', 'www.mediashopuk.com/17', 'www.mediashopuk.com/175', 'www.mediashopuk.com/19', 'www.mediashopuk.com/29', 'www.mediashopuk.com/3', 'www.mediashopuk.com/33', 'www.mediashopuk.com/37', 'www.mediashopuk.com/42', 'www.mediashopuk.com/45', 'www.mediashopuk.com/5', 'www.mediashopuk.com/52', 'www.mediashopuk.com/58', 'www.mediashopuk.com/86', 'www.mediashopuk.com/9', 'www.mrhightech.com/102', 'www.mrhightech.com/117', 'www.mrhightech.com/122', 'www.mrhightech.com/166', 'www.mrhightech.com/167', 'www.mrhightech.com/175', 'www.mrhightech.com/25', 'www.mrhightech.com/27', 'www.mrhightech.com/29', 'www.mrhightech.com/30', 'www.mrhightech.com/39', 'www.mrhightech.com/4', 'www.mrhightech.com/41', 'www.mrhightech.com/45', 'www.mrhightech.com/46', 'www.mrhightech.com/51', 'www.mrhightech.com/53', 'www.mrhightech.com/57', 'www.mrhightech.com/58', 'www.mrhightech.com/72', 'www.mrhightech.com/87', 'www.mrhightech.com/88', 'www.odsi.co.uk/108', 'www.odsi.co.uk/137', 'www.odsi.co.uk/140', 'www.odsi.co.uk/30', 'www.odsi.co.uk/32', 'www.odsi.co.uk/40', 'www.odsi.co.uk/49', 'www.odsi.co.uk/63', 'www.ohc24.ch/261', 'www.ohc24.ch/265', 'www.ohc24.ch/344', 'www.ohc24.ch/444', 'www.ohc24.ch/481', 'www.ohc24.ch/513', 'www.ohc24.ch/782', 'www.ohc24.ch/798', 'www.ohc24.ch/817', 'www.ohc24.ch/821', 'www.softwarecity.ca/1598', 'www.xpcpro.com/14', 'www.xpcpro.com/17', 'www.xpcpro.com/187', 'www.xpcpro.com/2', 'www.xpcpro.com/205', 'www.xpcpro.com/30', 'www.xpcpro.com/38', 'www.xpcpro.com/6', 'www.xpcpro.com/64', 'www.xpcpro.com/7', 'www.xpcpro.com/77', 'www.xpcpro.com/81', 'www.xpcpro.com/92']
will_keep_brands2=['unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'yiynova', 'yiynova', 'lucoms', 'unbranded/generic', 'unbranded/generic', 'neovo', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'no_comp', 'unbranded/generic', 'unbranded/generic', 'no_comp', 'ag neovo', 'unbranded/generic', 'no_comp', 'sensormatic', 'unbranded/generic', 'no_comp', 'eurosys', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'no_comp', 'unbranded/generic', 'belinea', 'unbranded/generic', 'unbranded/generic', 'silicon graphics', 'yusmart', 'unbranded/generic', 'emprex', 'videoseven', 'unbranded/generic', 'no_comp', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'belinea', 'unbranded/generic', 'unbranded/generic', 'no_comp', 'unbranded/generic', 'unbranded/Generic', 'iiyama', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generuc', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'matrox', 'no_comp', 'unbranded/generic', 'unbranded/generic', 'lyntek', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'funtica', 'unbranded/generic', 'unbranded/generic', 'bosch', 'unbranded/generic', 'silicon graphics', 'vibrant', 'skyport', 'unbranded/generic', 'unbranded/generic', 'no_comp', 'unbranded/generic', 'unbranded/generic', 'viglen', 'belinea', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'no_comp', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'alpha touch', 'unbranded/generic', 'medion', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'apple', 'unbranded/generic', 'digimate', 'unbranded/generic', 'jetway', 'unbranded/generic', 'unbranded/generic', 'aview', 'unbranded/generic', 'quanta', 'unbranded/generic', 'brilliance', 'lyntek', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'tenvis', 'sharp', 'advent', 'advent', 'unbranded/generic', 'hp', 'unbranded/generic', 'envision', 'unbranded/generic', 'megavision', 'unbranded/generic', 'unbranded/generic', 'bmw', 'unbranded/generic', 'belinea', 'nec', 'ikegami', 'unbranded/generic', 'vusys', 'unbranded/generic', 'unbranded/generic', 'harsper', 'unbranded/generic', 'unbranded/generic', 'angel', 'unbranded/generic', 'unbranded/generic', 'rackmux', 'unbranded/generic', 'unbranded/generic', 'envision', 'unbranded/generic', 'neova', 'yusmart', 'digimate', 'skyport', 'no_comp', 'unbranded/generic', 'belinea', 'unbranded/generic', 'belinea', 'austin hughes', 'belinea', 'unbranded/generic', 'digimate', 'hyvision', 'unbranded/generic', 'no_comp', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'iiyama', 'lucoms', 'no_comp', 'advent', 'unbranded/generic', 'tresor', 'unbranded/generic', 'unbranded/generic', 'iiyama', 'belinea', 'advent', 'sharp', 'unbranded/generic', 'viewsonic', 'medion', 'unbranded/generic', 'chunghwa', 'chatsworth', 'unbranded/generic', 'unbranded/generic', 'sharp', 'hannspree', 'hannspree', 'hannspree', 'hannspree', 'hannspree', 'hannspree', 'eaton', 'wasp', 'sharp', 'silicon graphics', 'sharp', 'envision', 'sunbrite', 'unbranded/generic', 'unbranded/generic', 'sharp', 'unbranded/generic', 'unbranded/generic', 'envision', 'envision', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'sharp', 'unbranded/generic', 'unbranded/generic', 'hitachi', 'sharp', 'hp', 'envision', 'unbranded/generic', 'hp', 'unbranded/generic', 'unbranded/generic', 'envision', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'envision', 'westinghouse', 'unbranded/generic', 'samsung', 'unbranded/generic', 'sharp', 'unbranded/generic', 'sharp', 'unbranded/generic', 'elite', 'unbranded/generic', 'zentview', 'unbranded/generic', 'unbranded/generic', 'sharp', 'envision', 'apple', 'knotron', 'unbranded/generic', 'unbranded/generic', 'sharp', 'bestech', 'unbranded/generic', 'unbranded/generic', 'gateway', 'hitachi', 'preh', 'crystalpro', 'balance', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'unbranded/generic', 'syncmaster', 'sva', 'unbranded/generic', 'starlogic', 'kogan', 'sunbrite', 'eizi', 'partnertech', 'puriton', 'kogan', 'sunbrite', 'sunbrite', 'sunbrite', 'sunbrite', 'sunbrite', 'sunbrite', 'sunbrite', 'hannspree', 'hannspree', 'hannspree', 'hannspree', 'hannspree', 'sharp', 'tv one', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'tv one', 'sharp', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'wortmann', 'sharp', 'nec', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'sharp', 'star micronics', 'neovo', 'roline', 'rotronic', 'rotronic', 'roline', 'roline', 'star micronics', 'intellinet', 'roline', 'sharp', 'neovo', 'neovo', 'neovo', 'neovo', 'neovo', 'neovo', 'neovo', 'neovo', 'neovo', 'neovo', 'neovo', 'neovo', 'neovo']

for i in range(0,len(will_keep2)):
    brand_dict[will_keep2[i]]=will_keep_brands2[i]

items=list(brand_dict.keys())
for item in items:
    if brand_dict[item]=="to_delete":
        del brand_dict[item]
    elif brand_dict[item]=="unbranded_generuc":
        brand_dict[item]="unbranded/generic"
    elif brand_dict[item]=="unbranded/generuc":
        brand_dict[item]="unbranded/generic"
    elif brand_dict[item]=="unbranded/Generic":
        brand_dict[item]="unbranded/generic"
    elif brand_dict[item]=="envision monitors":
        brand_dict[item]="envision"
    elif "elif" in brand_dict[item]:
        brand_dict[item]="unbranded/generic"
for item in brand_dict.keys():
    f = open("2013_monitor_specs/"+item+".json")
    data = json.load(f)
    if brand_dict[item]=="unbranded/generic":
        if "condition" in data and "\nBrand:" in data["condition"]:
            cand_brand=data["condition"].split("\nBrand:")[1].split("\n")[1].lower()
            if cand_brand=="hanns.g":
                cand_brand="hannspree"
            brand_dict[item]=cand_brand
brand_dict["www.best-deal-items.com/2206"]="chassis plans"
brand_dict["catalog.com/323"]="sharp"
brand_dict["catalog.com/397"]="sharp"
brand_dict["www.best-deal-items.com/1166"]="boscam"
brand_dict["www.best-deal-items.com/1626"]="sharp"
brand_dict["www.best-deal-items.com/1953"]="alpha touch"
brand_dict["ce.yikus.com/264"]="envision"
brand_dict["www.best-deal-items.com/854"]="acer"
brand_dict["www.ebay.com/22569"]="first"
brand_dict["www.ebay.com/12114"]="sharp"
brand_dict["www.best-deal-items.com/1115"]="xo vision"
brand_dict["www.best-deal-items.com/1656"]="faytech"
brand_dict["www.best-deal-items.com/906"]="videoseven"
brand_dict["www.best-deal-items.com/1481"]="rackmux"
brand_dict["www.best-deal-items.com/2213"]="videoseven"
brand_dict["www.best-deal-items.com/2170"]="alpha touch"
brand_dict["www.best-deal-items.com/2340"]="videoseven"
brand_dict["www.best-deal-items.com/624"]="videoseven"
brand_dict["www.best-deal-items.com/2260"]="totevision"
brand_dict["www.ebay.com/20565"]="auo"
brand_dict["www.ebay.com/21893"]="pge"
brand_dict["www.best-deal-items.com/1163"]="auo"
brand_dict["www.ebay.com/14697"]="ag neovo"
brand_dict["www.best-deal-items.com/396"]="edge10"
brand_dict["www.best-deal-items.com/2151"]="nec"
brand_dict["www.best-deal-items.com/787"]="tatung"
brand_dict["www.best-deal-items.com/2669"]="primera"
brand_dict["www.best-deal-items.com/1679"]="polycom"
brand_dict["www.best-deal-items.com/1443"]="yuraku"
brand_dict["www.best-deal-items.com/2776"]="cibox"
brand_dict["www.best-deal-items.com/918"]="dell"
brand_dict["www.best-deal-items.com/305"]="aoc"
brand_dict["www.best-deal-items.com/998"]="benq"
brand_dict["www.best-deal-items.com/1630"]="sun microsystems"
brand_dict["www.best-deal-items.com/1319"]="digimate"
brand_dict["www.best-deal-items.com/1096"]="unbranded/generic"
items=list(brand_dict.keys())
for item in items:
    if brand_dict[item]=="no_comp":
        del brand_dict[item]
    elif brand_dict[item]=="N.A":
        del brand_dict[item]

del brand_dict["www.best-deal-items.com/1821"]
del brand_dict["www.ebay.com/23179"]
del brand_dict["www.best-deal-items.com/1715"]
del brand_dict["www.ebay.com/20767"]
del brand_dict["www.best-deal-items.com/1064"]
#del brand_dict["www.best-deal-items.com/959"]
del brand_dict["www.best-deal-items.com/2600"]
#del brand_dict["www.best-deal-items.com/197"]
#del brand_dict["www.best-deal-items.com/2704"]
#del brand_dict["www.pcconnection.com/1685"]
#del brand_dict["www.pcconnection.com/1579"]
#del brand_dict["www.pcconnection.com/2676"]
del brand_dict["www.pcconnection.com/1030"]
#del brand_dict["www.pcconnection.com/2170"]
del brand_dict["www.pcconnection.com/2053"]
#del brand_dict["www.pcconnection.com/1171"]
#del brand_dict["www.pcconnection.com/1864"]
#del brand_dict["www.pcconnection.com/2992"]
#del brand_dict["www.pcconnection.com/2948"]
#del brand_dict["www.pcconnection.com/1114"]
#del brand_dict["www.pcconnection.com/3739"]
del brand_dict["www.pcconnection.com/1656"]
#del brand_dict["www.pcconnection.com/1251"]
#del brand_dict["www.pcconnection.com/1389"]
del brand_dict["www.best-deal-items.com/269"]
del brand_dict["www.best-deal-items.com/1692"]
del brand_dict["www.best-deal-items.com/175"]
del brand_dict["www.best-deal-items.com/1546"]
#del brand_dict["www.best-deal-items.com/103"]
del brand_dict["www.ebay.com/21507"]
del brand_dict["www.ebay.com/21785"]
#del brand_dict["www.pcconnection.com/2723"]
#del brand_dict["www.pcconnection.com/3341"]
del brand_dict["www.pcconnection.com/3750"]
#del brand_dict["www.pcconnection.com/2480"]
#del brand_dict["www.pcconnection.com/2083"]
#del brand_dict["www.best-deal-items.com/646"]
#del brand_dict["www.ebay.com/19543"]
del brand_dict["www.best-deal-items.com/943"]
del brand_dict["www.best-deal-items.com/2049"]
del brand_dict["www.best-deal-items.com/1560"]
del brand_dict["www.best-deal-items.com/2588"]
#del brand_dict["www.best-deal-items.com/935"]
del brand_dict["www.best-deal-items.com/988"]
del brand_dict["www.best-deal-items.com/1931"]
del brand_dict["www.best-deal-items.com/1376"]
del brand_dict["www.best-deal-items.com/725"]
del brand_dict["www.best-deal-items.com/461"]
del brand_dict["www.best-deal-items.com/1834"]
del brand_dict["www.best-deal-items.com/2127"]
del brand_dict["www.best-deal-items.com/2680"]
del brand_dict["www.best-deal-items.com/320"]
#del brand_dict["www.best-deal-items.com/2627"]
del brand_dict["www.best-deal-items.com/1550"]
del brand_dict["www.best-deal-items.com/335"]
del brand_dict["www.best-deal-items.com/627"]
del brand_dict["www.best-deal-items.com/2100"]
del brand_dict["www.best-deal-items.com/606"]
del brand_dict["www.best-deal-items.com/2119"]
del brand_dict["www.best-deal-items.com/2576"]
#del brand_dict["www.best-deal-items.com/1004"]
#del brand_dict["www.best-deal-items.com/1845"]
del brand_dict["www.best-deal-items.com/1846"]
#del brand_dict["www.best-deal-items.com/1108"]
del brand_dict["www.best-deal-items.com/2026"]
del brand_dict["www.best-deal-items.com/2484"]
#del brand_dict["www.best-deal-items.com/54"]
del brand_dict["www.best-deal-items.com/1142"]
del brand_dict["www.best-deal-items.com/287"]
del brand_dict["www.best-deal-items.com/895"]
del brand_dict["www.best-deal-items.com/302"]
del brand_dict["www.best-deal-items.com/1814"]
del brand_dict["www.best-deal-items.com/884"]
del brand_dict["www.best-deal-items.com/2273"]
del brand_dict["www.best-deal-items.com/1687"]
del brand_dict["www.best-deal-items.com/2682"]
del brand_dict["www.best-deal-items.com/2"]
del brand_dict["www.best-deal-items.com/2303"]
del brand_dict["www.best-deal-items.com/486"]
del brand_dict["www.best-deal-items.com/2476"]
#del brand_dict["www.best-deal-items.com/669"]
del brand_dict["www.best-deal-items.com/2031"]
del brand_dict["www.best-deal-items.com/1164"]
del brand_dict["www.best-deal-items.com/366"]
del brand_dict["www.best-deal-items.com/2714"]
del brand_dict["www.best-deal-items.com/361"]
del brand_dict["www.best-deal-items.com/394"]
del brand_dict["www.best-deal-items.com/1861"]
del brand_dict["www.best-deal-items.com/793"]
del brand_dict["www.best-deal-items.com/2470"]
del brand_dict["www.best-deal-items.com/1410"]
del brand_dict["www.best-deal-items.com/1390"]
del brand_dict["www.best-deal-items.com/2268"]
del brand_dict["www.best-deal-items.com/2105"]
del brand_dict["www.ebay.com/23432"]
del brand_dict["www.ebay.com/21663"]
del brand_dict["www.ebay.com/9914"]
del brand_dict["www.ebay.com/14382"]
#del brand_dict["www.ebay.com/18030"]
del brand_dict["www.ebay.com/21210"]
del brand_dict["www.ebay.com/21909"]
del brand_dict["www.ebay.com/22939"]
del brand_dict["www.ebay.com/12174"]
del brand_dict["www.ebay.com/11510"]
#del brand_dict["www.best-deal-items.com/331"]
del brand_dict["www.makingbuyingeasy.co.uk/419"]

count=0
for item in sorted(list(brand_dict.keys())):
    item_to_id[item]=count
    count+=1
    
with open('brand_dict.pkl', 'wb') as handle:
    pickle.dump(brand_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('item_to_id.pkl', 'wb') as handle:
    pickle.dump(item_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
brands=set()
for i in brand_dict:
    brands.add(brand_dict[i])

#print("Our brands...")
#for i in sorted(list(brands)):
#    print(i)

for item in quick_fix.keys():
    if not type(quick_fix[item]) is list:
        quick_fix[item]=[quick_fix[item]]
a=["thinkvision", "accusync","accutouch","acer", "thinkpad","syncmaster","prodisplay","pavilion","promaster","apple","flatron","dell","compaq", "acer", "aoc", "playstation","latitude","hp","asus","acerview", "asuspro", "benq","envy", "multisync", "radiforce","precision", "sparta", "brilliance", "cintiq", "deluxepro","dreamcolor","value","entuitive","foris","elitebook","intellitouch","mdcg","kingtee","prolite","smartbuy","sympodium","trutouch","viewsonic"]
del quick_fix["hanns g"]
del quick_fix["allen bradley"]

final_brand_dict = {**quick_fix, **brand_dict2}
final_brand_dict["envision monitors"]=["envision", "envision monitors"]
final_brand_dict["rockwell"]=["rockwell", "allen bradley", "allen-bradley"]
final_brand_dict["lyntek"]=["lynteck", "lyntek"]
fixes=set()

"""
Step 3: Model cleaning
"""
print("Phase 3: In this step we determine the models. This part follows four steps, with some cleaning involved. The first step consists of rules to extract candidate models strings from attributes. Then a voting process, where the most frequent already extracted models from the first step are used to guide the identification of models on page titles. The third steps applies rules for detecting models from the items, with a focus on page titles. The last step returns to the voting scheme to further extract model strings. (This should not take more than a minute or two.)")

data_keys=set()
unknown_count=0
problems=set()
model_dict=dict()
from_voting=set()

print("First part: Basic rule-based model extraction from explicit attributes")
#The first part of our solution for model assignation, also follows rules, to extract from candidate attributes (model, product_model, ...) only.
bdd=copy.deepcopy(brand_dict)
brand_to_models=dict()
for item in sorted(list(bdd.keys())):
    f = open("2013_monitor_specs/"+item+".json")
    data = json.load(f)
    search_for="nothing"
    if not bdd[item] in final_brand_dict:
        final_brand_dict[bdd[item]]=[bdd[item]]
    bd2=[x.lower() for x in final_brand_dict[bdd[item]]]
    bd2.append(bdd[item].lower())
    bd2=list(set(bd2))
    if "model" in data and data["model"][0:2]==": ":
        data["model"]=data["model"][2:]
    if "model" in data and data["model"].count("-")>1:
        data["model"]=data["model"].replace("-"," ")
    if "condition" in data and "\nModel:" in data["condition"]:
       cand_model=data["condition"].split("\nModel:")[1].split("\n")[1].lower()
       if not bdd[item] in brand_to_models:
           brand_to_models[bdd[item]]=set()
       brand_to_models[bdd[item]].add(cand_model)
       model_dict[item]=cand_model
    elif "model" in data and (str(data["model"]).lower() in str(data["<page title>"]).lower() or not " " in str(data["model"]).lower()):
       search_for="model"
    elif "product model" in data and (str(data["product model"]).lower() in str(data["<page title>"]).lower() or not " " in str(data["product model"]).lower()):
       search_for="product model"
    elif "product name" in data and (str(data["product name"]).lower() in str(data["<page title>"]).lower() or not " " in str(data["product name"]).lower()):
       search_for="product name"
    elif "mpn" in data and str(data["mpn"]).lower() in str(data["<page title>"]).lower():
       search_for="mpn"
    elif "mfr part number" in data and str(data["mfr part number"]).lower() in str(data["<page title>"]).lower():
       search_for="mfr part number"
    elif "model name" in data and (str(data["model name"]).lower() in str(data["<page title>"]).lower() or not " " in str(data["model name"]).lower()):
        search_for="model name"
    elif "series" in data and str(data["series"]).lower() in str(data["<page title>"]).lower():
       search_for="series"
    elif "model number" in data and str(data["model number"]).lower() in str(data["<page title>"]).lower():
       search_for="model number"
    elif "â model number" in data and str(data["â model number"]).lower() in str(data["<page title>"]).lower():
       search_for="â model number"
    elif "specifications" in data and str(data["specifications"]).lower() in str(data["<page title>"]).lower():
       search_for="specifications"
    elif "specs" in data and str(data["specs"]).lower() in str(data["<page title>"]).lower():
       search_for="specs"
    if search_for!="nothing":
        cand_model=str(data[search_for]).lower().replace("\n","").replace(":","").replace("'","").replace("\"","").strip()
        if (not " " in cand_model) and (cand_model not in bd2) and any(i.isdigit() for i in cand_model) and (cand_model not in a):
            for word in bd2:
                cand_model=cand_model.replace(word,"")
            cand_model=cand_model.replace("\n","").strip()
            if all(i.isdigit() for i in cand_model):
                if int(cand_model)<30:
                    model_dict[item]="unknown"
                elif len(cand_model)>0 and cand_model not in stopwords and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))) and not all(i.isdigit() for i in cand_model):
                    if not bdd[item] in brand_to_models:
                        brand_to_models[bdd[item]]=set()
                    brand_to_models[bdd[item]].add(cand_model)
                    model_dict[item]=cand_model
            elif len(cand_model)>0 and cand_model not in stopwords and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))) and not all(i.isdigit() for i in cand_model):
                model_dict[item]=cand_model
                if not bdd[item] in brand_to_models:
                    brand_to_models[bdd[item]]=set()
                brand_to_models[bdd[item]].add(cand_model)
        elif cand_model[0]=="[" and cand_model[len(cand_model)-1]=="]":
            for word in bd2:
                cand_model=cand_model.replace(word,"")
            for word in a:
                cand_model=cand_model.replace(word,"")
            cand_model= cand_model.strip('][').split(', ')[1].split("\\n")[0].replace("\n","").replace("'","").replace("\"","").replace("\"","").strip()
            if len(cand_model)>0 and cand_model not in stopwords and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))) and not all(i.isdigit() for i in cand_model):
                model_dict[item]=cand_model
                if not bdd[item] in brand_to_models:
                    brand_to_models[bdd[item]]=set()
                brand_to_models[bdd[item]].add(cand_model)        
        elif any(x in cand_model for x in ["x series","black tune","rog swift"]):
            for word in bd2:
                cand_model=cand_model.replace(word,"")
            for word in a:
                cand_model=cand_model.replace(word,"")
            cand_model= cand_model.replace("\n","").replace("'","").replace("\"","").replace("\"","").strip()
            if len(cand_model)>0 and cand_model not in stopwords and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))) and not all(i.isdigit() for i in cand_model):
                model_dict[item]=cand_model
                if not bdd[item] in brand_to_models:
                    brand_to_models[bdd[item]]=set()
                brand_to_models[bdd[item]].add(cand_model)
        elif len(cand_model.split(" "))==3 and all(x.isdigit() for x in cand_model.split(" ")):
            if len(cand_model)>0 and cand_model not in stopwords and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))) and not all(i.isdigit() for i in cand_model):
                model_dict[item]=cand_model
                if not bdd[item] in brand_to_models:
                    brand_to_models[bdd[item]]=set()
                brand_to_models[bdd[item]].add(cand_model)
        elif len(cand_model.split(" "))==2 and any(x in a for x in cand_model.split(" ")):
            for word in bd2:
                cand_model=cand_model.replace(word,"")
            for word in a:
                cand_model=cand_model.replace(word,"")
            cand_model=cand_model.replace("\n","").replace("'","").replace("\"","").replace("\"","").strip()
            if len(cand_model)>0 and cand_model not in stopwords and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))) and not all(i.isdigit() for i in cand_model):
                model_dict[item]=cand_model
                if not bdd[item] in brand_to_models:
                    brand_to_models[bdd[item]]=set()
                brand_to_models[bdd[item]].add(cand_model)
        elif len(cand_model.split(" "))>2 and any(x in a for x in cand_model.split(" ")) and any(i.isdigit() for i in cand_model.split(" ")[1]):
            for word in bd2:
                cand_model=cand_model.replace(word,"")
            for word in a:
                cand_model=cand_model.replace(word,"")
            cand_model=(cand_model.split(" ")[0]+" "+cand_model.split(" ")[1]).replace("\n","").replace("'","").replace("\"","").strip()
            if len(cand_model)>0 and cand_model not in stopwords and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))) and not all(i.isdigit() for i in cand_model):
                model_dict[item]=cand_model
                if not bdd[item] in brand_to_models:
                    brand_to_models[bdd[item]]=set()
                brand_to_models[bdd[item]].add(cand_model)
        elif len(cand_model.split(" "))>2 and any(x in a for x in [cand_model.replace(" led","").split(" ")[1]]) and any(i.isdigit() for i in cand_model.split(" ")[0]):
            cand_model=cand_model.replace(" led","")
            for word in bd2:
                cand_model=cand_model.replace(word,"")
            for word in a:
                cand_model=cand_model.replace(word,"")
            cand_model=(cand_model.split(" ")[0]+" "+cand_model.split(" ")[1]).replace("\n","").replace("'","").replace("\"","").strip()
            if len(cand_model)>0 and cand_model not in stopwords and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))) and not all(i.isdigit() for i in cand_model):
                model_dict[item]=cand_model
                if not bdd[item] in brand_to_models:
                    brand_to_models[bdd[item]]=set()
                brand_to_models[bdd[item]].add(cand_model)
        elif "model" in cand_model:
            cand_model= cand_model.split("model")[1].split(" ")[0].replace("\n","").replace("'","").replace("\"","").strip()
            if len(cand_model)>0 and cand_model not in stopwords and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))) and not all(i.isdigit() for i in cand_model):
                model_dict[item]=cand_model
                if not bdd[item] in brand_to_models:
                    brand_to_models[bdd[item]]=set()
                brand_to_models[bdd[item]].add(cand_model)



#The second part of our model assignation process uses the frequency of extracted models to guide the process. The idea is that models that have already been extracted for a brand, if found on the title, can be indicative of the model name.
print("Second part: Frequency-based model assignation (voting)")
model_freq=dict()
for item in model_dict.keys():
    if not model_dict[item] in model_freq:
        model_freq[model_dict[item]]=1
    else:
        model_freq[model_dict[item]]+=1

for item in sorted(list(bdd.keys())):
    f = open("2013_monitor_specs/"+item+".json")
    data = json.load(f)
    search_for="nothing"
    if not bdd[item] in final_brand_dict:
        final_brand_dict[bdd[item]]=[bdd[item]]
   
    bd2=[x.lower() for x in final_brand_dict[bdd[item]]]
    bd2.append(bdd[item].lower())
    bd2=list(set(bd2))
    if (not item in model_dict) or model_dict[item]=="unknown":
        model_chosen=False
        if bdd[item] in brand_to_models:
            inner_dict=dict() 
            for k in list(brand_to_models[bdd[item]]):
                inner_dict[k]=model_freq[k]
            for it in list(sorted(inner_dict, key=inner_dict.get, reverse=True)):
                if it in data["<page title>"].lower():
                    model_dict[item]=it
                    model_freq[it]+=1
                    model_chosen=True
                    from_voting.add(item)
                    break
            if not model_chosen:
                inner_dict=dict() 
                for k in list(brand_to_models[bdd[item]]):
                    inner_dict[k]=model_freq[k]
                for it in list(sorted(inner_dict, key=inner_dict.get, reverse=True)):
                    for key in data.keys():
                        if it in str(data[key]).lower():
                            model_dict[item]=it
                            model_freq[it]+=1
                            model_chosen=True
                            from_voting.add(item)
                            break
                    if model_chosen:
                        break


#The third part of our model assignation process refers to the page title only
print("Third part, rule-based assignation using the page title. It is done after the first voting, such that voting takes priority over it.")
for item in sorted(list(bdd.keys())):
    f = open("2013_monitor_specs/"+item+".json")
    data = json.load(f)
    search_for="nothing"
    if not bdd[item] in final_brand_dict:
        final_brand_dict[bdd[item]]=[bdd[item]]
    bd2=[x.lower() for x in final_brand_dict[bdd[item]]]
    bd2.append(bdd[item].lower())
    bd2=list(set(bd2))
    if (not item in model_dict) or model_dict[item]=="unknown":
        if any(x in bd2 for x in data["<page title>"].lower().replace("(","").replace(")","").split(" ")) or any([x in data["<page title>"].lower().replace("(","").replace(")","") for x in bd2]):
            brand_mention=""
            for word in bd2:
                if word in data["<page title>"].lower().replace("(","").replace(")","").split(" ") or word in data["<page title>"].lower().replace("(","").replace(")",""):
                    brand_mention=word
            cand_model=""
            string_to_search=data["<page title>"].lower().replace("(","").replace(")","").replace("\"","").split(brand_mention)[1]
            if len(string_to_search.split(" "))>1:
                string_to_search=string_to_search.split(" ")[1]
            else:
                string_to_search=data["<page title>"].lower().replace("(","").replace(")","").replace("\"","").split(" ")[0]
            if any(i.isdigit() for i in string_to_search):
                cand_model=string_to_search
            if len(cand_model)>0 and cand_model not in stopwords and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))) and not all(i.isdigit() for i in cand_model) and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))):
                model_dict[item]=cand_model
                if not bdd[item] in brand_to_models:
                    brand_to_models[bdd[item]]=set()
                brand_to_models[bdd[item]].add(cand_model)
            else:
                strings_to_search=data["<page title>"].lower().replace("(","").replace(")","").replace("\"","").split(" ")
                longest_with_digits_and_letters=-1
                length_wdal=-1
                longest_with_digits=-1
                length_wd=-1
                for k in range(0,len(strings_to_search)):
                    sts=strings_to_search[k]
                    if any(i.isdigit() for i in sts):
                        if all(i.isdigit() for i in sts.replace("-","").replace(".","").replace(",","").replace("inch","").replace("\"","").replace(":","").replace("²","")) and len(sts)>length_wd:
                            length_wd=len(sts)
                            longest_with_digits=k
                        elif len(sts)>length_wdal:
                            length_wdal=len(sts)
                            longest_with_digits_and_letters=k
                if length_wdal>=3 and longest_with_digits_and_letters>=0 and ("\"" not in strings_to_search[longest_with_digits_and_letters]):
                    cand_model=strings_to_search[longest_with_digits_and_letters]
                    if len(cand_model)>0 and cand_model not in stopwords and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))) and not all(i.isdigit() for i in cand_model):
                        model_dict[item]=cand_model
                        if not bdd[item] in brand_to_models:
                            brand_to_models[bdd[item]]=set()
                        brand_to_models[bdd[item]].add(cand_model)
                elif length_wd>=3 and longest_with_digits>=0:
                    cand_model=strings_to_search[longest_with_digits]
                    if len(cand_model)>0 and cand_model not in stopwords and not(len(cand_model.split("."))=="2" and all(i.isdigit() for i in cand_model.replace(".",""))) and not all(i.isdigit() for i in cand_model):
                        model_dict[item]=cand_model
                        if not bdd[item] in brand_to_models:
                            brand_to_models[bdd[item]]=set()
                        brand_to_models[bdd[item]].add(cand_model)

#The final part of our solution refers once again to voting, but this time using the items also identified from the page title
print("Final part, assignation again based on voting. This time using the models assigned from rules and page titles")
model_freq=dict()
for item in model_dict.keys():
    if not model_dict[item] in model_freq:
        model_freq[model_dict[item]]=1
    else:
        model_freq[model_dict[item]]+=1
for item in bdd.keys():
    f = open("2013_monitor_specs/"+item+".json")
    data = json.load(f)
    search_for="nothing"
    if not bdd[item] in final_brand_dict:
        final_brand_dict[bdd[item]]=[bdd[item]]
   
    bd2=[x.lower() for x in final_brand_dict[bdd[item]]]
    bd2.append(bdd[item].lower())
    bd2=list(set(bd2))
    if (not item in model_dict) or model_dict[item]=="unknown":
        model_chosen=False
        if bdd[item] in brand_to_models: 
            inner_dict=dict()
            for k in list(brand_to_models[bdd[item]]):
                inner_dict[k]=model_freq[k]
            for it in list(sorted(inner_dict, key=inner_dict.get, reverse=True)):
                if it in data["<page title>"].lower():
                    model_dict[item]=it
                    model_freq[it]+=1
                    from_voting.add(item)
                    model_chosen=True
                    break
            if not model_chosen:
                inner_dict=dict() 
                for k in list(brand_to_models[bdd[item]]):
                    inner_dict[k]=model_freq[k]
                for it in list(sorted(inner_dict, key=inner_dict.get, reverse=True)):
                    for key in data.keys():
                        if it in str(data[key]).lower():
                            model_dict[item]=it
                            model_freq[it]+=1
                            model_chosen=True
                            from_voting.add(item)
                            break
                    if model_chosen:
                        break
        if not model_chosen:
            model_dict[item]="unknown"
            unknown_count+=1

#We complement with some data cleaning
model_dict["www.officedepot.com/123"]="trutouch 460"
model_dict["www.officedepot.com/265"]="trutouch 650"
model_dict["www.officedepot.com/177"]="1700"
model_dict["ce.yikus.com/283"]="m lynx 10"
model_dict["ce.yikus.com/108"]="106960"
model_dict["www.ebay.com/18659"]="al1711"
model_dict["www.best-deal-items.com/1971"]="al1916"
unknown_count-=5
for item in brand_dict.keys():
    if brand_dict[item]=="apple" and model_dict[item]=="unknown":
        f = open("2013_monitor_specs/"+item+".json")
        data = json.load(f)
        for key in data.keys():
            if "thunderbolt" in str(data[key]).lower() or "thunderbold" in str(data[key]).lower():
                model_dict[item]="thunderbolt"
                unknown_count-=1
                break
            elif "cinema" in str(data[key]).lower():
                model_dict[item]="cinema display"
                unknown_count-=1
                break
            elif "studio display" in str(data[key]).lower() or "studio hd display" in str(data[key]).lower():
                model_dict[item]="studio display"
                unknown_count-=1
                break

for item in sorted(list(bdd.keys())):
    f = open("2013_monitor_specs/"+item+".json")
    data = json.load(f)
    if not bdd[item] in final_brand_dict:
        final_brand_dict[bdd[item]]=[bdd[item]]   
    bd2=[x.lower() for x in final_brand_dict[bdd[item]]]
    bd2.append(bdd[item].lower())
    bd2=list(set(bd2))
    if model_dict[item]!="unknown":
        if bdd[item] in brand_to_models: 
            inner_dict=dict()
            for k in list(brand_to_models[bdd[item]]):
                inner_dict[k]=model_freq[k]
            for it in list(sorted(inner_dict, key=inner_dict.get, reverse=True)):
                if (it in data["<page title>"].lower()) and it!=model_dict[item] and model_freq[it]>model_freq[model_dict[item]]:
                    model_freq[model_dict[item]]-=1
                    model_dict[item]=it
                    model_freq[it]=+1
                    break
         
brand_to_items=dict()
for item in brand_dict:
    if not brand_dict[item] in brand_to_items:
        brand_to_items[brand_dict[item]]=set([item])
    else:
        brand_to_items[brand_dict[item]].add(item)
#print(sorted(list(set([model_dict[k] for k in model_dict.keys()]))))
model_freq=dict()
for item in brand_dict.keys():
    if not model_dict[item] in model_freq:
        model_freq[model_dict[item]]=1
    else:
        model_freq[model_dict[item]]+=1
model_dict["www.ebay.com/9449"]="301 sparta"
model_dict["www.ebay.com/23627"]="301 sparta"
model_dict["www.ebay.com/23498"]="301 sparta"
model_dict["www.ebay.com/19935"]="301 sparta"
model_dict["www.ebay.com/18648"]="301 sparta"
with open('model_dict.pkl', 'wb') as handle:
    pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""
Step 4: Entity Resolution
"""
print("Phase 4: Entity Resolution. In this step we compare all items in the same brand, and with similar models. (It might take to run around 10 minutes, depending on the machine. We should improve this)")

output=pd.DataFrame(columns=['left_spec_id', 'right_spec_id'])#create a new dataframe
brand_models=dict()
to_exclude_brands=["aver","omron"]
to_exclude_models=["1080p", "700p","700g", "250cd/m2","21.3"] #False positives (common)
for item in brand_dict:
    if not brand_dict[item] in brand_models and not brand_dict[item] in to_exclude_brands:
        brand_models[brand_dict[item]]=dict()
    if model_dict[item]!="unknown" and not brand_dict[item] in to_exclude_brands and not model_dict[item] in to_exclude_models:
        if not model_dict[item] in brand_models[brand_dict[item]]:
            brand_models[brand_dict[item]][model_dict[item]]=0    
        brand_models[brand_dict[item]][model_dict[item]]+=1
counter=0
added=set()
brand_counter=0
brand_max=len(brand_models.items())
for brand in sorted(brand_models.items(), key=lambda kv: len(kv[1]), reverse=True):
    brand=brand[0]
    brand_counter+=1
    print("Evaluating brand: "+brand+" (#"+str(brand_counter)+"/"+str(brand_max)+") (sorted by number of models per brand)")
    if True: #We could add some conditions here... (e.g. not considering unbranded/generic)
        if True: #Nothing added ATM.
            #print(str(counter))
            #counter+=1
            #print(brand+": "+str(sorted(brand_models[brand].items(), key=lambda kv: kv[1], reverse=True)))
            for k in sorted(brand_models[brand].items(), key=lambda kv: kv[1], reverse=True):
                collection1=[]
                for i in model_dict:
                    if brand_dict[i]==brand and (model_dict[i]==k[0] or model_dict[i] in k[0] or k[0] in model_dict[i]):
                        collection1.append(i)
                if len(collection1)>1:
                    for i in list(itertools.combinations(collection1, 2)):
                        if not i[0]+"_"+i[1] in added:
                            output=output.append({'left_spec_id': i[0].replace("/","//"), 'right_spec_id': i[1].replace("/","//")}, ignore_index=True)
                            added.add(i[0]+"_"+i[1])

output.to_csv("output.csv", sep=',',encoding='utf-8', index=False)
