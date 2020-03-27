'''Global constants, and file paths'''
import os,re

# Change the path according to your system
embed_path = os.path.expanduser('~') + "/git-workspace/glove/generic/glove.840B.300d.txt"    #file containing glove embedding
core_nlp_url = 'http://localhost:9000'     #local host url of standford corenlp server
allFolderPath = os.path.expanduser('~')+"/git-workspace/amr_data/e25/data/alignments/split/"
#allFolderPath = os.path.expanduser('~')+"/git-workspace/learnlab/data/e25/data/amrs/split/"
resource_folder_path  = os.path.expanduser('~')+"/git-workspace/amr_data/e25/"
frame_folder_path = resource_folder_path+"/data/frames/propbank-frames-xml-2016-03-08/"
vallex_file_path = os.path.expanduser('~')+"/git-workspace/mrp_data/vallex_en.xml"
semi_folder_path = os.path.expanduser('~')+"/git-workspace/mrp_data/1214/etc/"
dm_mwe_file= os.path.expanduser('~')+"/git-workspace/mrp_data/dm.joints"
psd_mwe_file= os.path.expanduser('~')+"/git-workspace/mrp_data/psd.joints"
#allFolderPath = os.path.expanduser('~')+"/Data/amr_annotation_r2/data/alignments/split"
#resource_folder_path  = os.path.expanduser('~')+"/Data/amr_annotation_r2/"
#frame_folder_path = resource_folder_path+"data/frames/propbank-frames-xml-2016-03-08/"
have_org_role = resource_folder_path+"have-org-role-91-roles-v1.06.txt"   #not used
have_rel_role = resource_folder_path+"have-rel-role-91-roles-v1.06.txt"   #not used
morph_verbalization = resource_folder_path+"morph-verbalization-v1.01.txt"  #not used
verbalization =  resource_folder_path+"verbalization-list-v1.06.txt"


PAD = 0
UNK = 1

# BERT_TOKENS
BERT_PAD = '[PAD]'
BERT_PAD_INDEX = 0
BERT_SEP = '[SEP]'
BERT_SEP_INDEX = 102
BERT_CLS= '[CLS]'
BERT_CLS_INDEX = 101


PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
NULL_WORD = ""
UNK_WIKI = '<WIKI>'
MWE_END = "<MWE_END>"
Special = [NULL_WORD,UNK_WORD,PAD_WORD]
#Categories
Rule_Frame = "Frame"
Rule_Constant = "Constant"
Rule_String = "String"
Rule_Concept = "Concept"
Rule_Comp = "COMPO"
Rule_Num = "Num"
Rule_Re = "Re"    #corenference
Rule_Ner = "Ner"
Rule_B_Ner = "B_Ner"
Rule_Other = "Entity"
Other_Cats = {"person","thing",}
COMP = "0"
Rule_All_Constants = [Rule_Num,Rule_Constant,Rule_String,Rule_Ner]
Splish = "$£%%££%£%£%£%"
Rule_Basics = Rule_All_Constants + [Rule_Frame,Rule_Concept,UNK_WORD,BOS_WORD,EOS_WORD,NULL_WORD,PAD_WORD]

RULE = 0
HIGH = 1
LOW = 2

RE_FRAME_NUM = re.compile(r'-\d\d$')
RE_COMP = re.compile(r'_\d$')
end= re.compile(".txt\_[a-z]*")
epsilon = 1e-8

TXT_WORD = 0
TXT_LEMMA = 1
TXT_POS = 2
TXT_NER = 3


# for AMR
AMR_CAT = 0
AMR_LE = 1
AMR_NER = 2
AMR_AUX = 2
AMR_LE_SENSE = 3
AMR_SENSE = 3
AMR_CAN_COPY = 4

# for DM
DM_POS = 0
DM_CAT = 1
DM_SENSE = 2
DM_LE = 3
DM_CAN_COPY = 4
DM_LE_CAN_COPY = 5

# For EDS
EDS_CAT = 0
EDS_TAG = 1
EDS_LE = 2
EDS_AUX = 3
EDS_CARG= 4
EDS_CAN_COPY = 5

# for PSD
PSD_POS = 0
PSD_LE = 1
PSD_SENSE = 2
PSD_CAN_COPY = 3


# index in sourceBatch
TOK_IND_SOURCE_BATCH=0
LEM_IND_SOURCE_BATCH=1
POS_IND_SOURCE_BATCH=2
NER_IND_SOURCE_BATCH=3
MWE_IND_SOURCE_BATCH=4
ANCHOR_IND_SOURCE_BATCH=5
TOTAL_INPUT_SOURCE=6

C_IND_SOURCE_BATCH=6
R_IND_SOURCE_BATCH=7
TRIPLE_IND_SOURCE_BATCH=8
CN_IND_SOURCE_BATCH=9
CC_IND_SOURCE_BATCH=10

threshold = 5


