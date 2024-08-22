import sys
import os
import json
import requests
from typing import List, Tuple
from utils import Model
import readability
import simple_icd_10 as icd
from html.parser import HTMLParser
import yaml
import importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import alfworld
import alfworld_runs
import alfworld.alfworld.agents.environment
from alfworld_runs import *

from importlib import reload
from typing import Dict, List, Any
from fhir.resources.communication import Communication
from fhir.resources.composition import Composition
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.fhirtypes import DateTime, ReferenceType
from fhir.resources.reference import Reference

class HTMLFilter(HTMLParser):
    text = ""
    def handle_data(self, data):
        self.text += data

class Model:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, orig_codes: dict, prompt: str, context: str, input_letter: str) -> str:
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a Primary Care Physician, and your job is to edit and refine a patient-friendly letter to ensure that it is medically and factually accurate, in an appropriate tone and language, and matching the reading level of a 6th grader if it was measured using the Flesch–Kincaid readability test. Here is the input letter you should be editing: \n\n{input_letter}"},  # system prompt
                {"role": "user", "content": f"{prompt}\n\n{context}, here is the input letter: \n\n{input_letter} and here are the original ICD-10 codes you should be including: \n\n{orig_codes}. Please make sure that the original ICD-10 codes match the ones in the patient-friendly letter."},  # user prompt
            ],
            "temperature": 0.7
        }
        response = requests.post(MD_AI_URL, json=body, headers=MD_AI_HEADERS)
        response_data = response.json()
        choices = response_data.get('response', {}).get('choices', [])
        if choices:
            text = choices[0].get('message', {}).get('content', '')
            return text
        return ""

model_instance = Model("gpt-4o")

#loading prompt file
FOLDER = 'alfworld_runs/prompts'
PROMPT_FILE = 'alfworld_3prompts.json'

with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

#URL and headers for the md.ai endpoint
MD_AI_URL = 'https://chat.md.ai/api/openai/chat/completions'
MD_AI_HEADERS = {
    'Content-Type': 'application/json',
    'x-access-token': '{TOKEN}' #alter token to personal token
}
FHIR_URL = "https://hackathon.siim.org/fhir-r4/"
FHIR_HEADERS = {
  'apikey': '{API_KEY}',
  'Content-Type': 'application/json'
}

#prompting LLM 

def llm(prompt: str, model: Model, stop: List[str] = ["\n"], num_versions: int = 5) -> List[Dict[str, Any]]:
    versions = []

    for i in range(num_versions):
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a Primary Care Physician, and your job is to generate a patient-friendly letter that is medically and factually accurate, in an appropriate tone and language, matching the reading level of a 6th grader if it was measured using the Flesch–Kincaid readability test"},  # system prompt
                {"role": "user", "content": prompt},  #user prompt
            ],
            "temperature": i * 0.3 #play around with this. higher, more variance
        }
        response = requests.post(MD_AI_URL, json=body, headers=MD_AI_HEADERS)
        #print(response.content)
        response_data = response.json()
        choices = response_data.get('response', {}).get('choices', [])
        if choices:
            text = choices[0].get('message', {}).get('content', '')
            #if "ICD-10" in text:
            if len(text.strip()) >= 5:
                version_data = {
                    'letter' : text, 'readability' : 0, 'accuracy' : 0 #placeholders for accuracy and readability values
                }
                versions.append(version_data)
    return versions

def check_readability(text: str) -> float:
    #Flesch-Kincaid readability score - score of 5.0-6.0 indicates 6th grade level
    readability_result = readability.getmeasures(text, lang='en')
    print("KINCAID:", readability_result['readability grades']['Kincaid'])
    readability_score = readability_result['readability grades']['Kincaid']
    #need to normalise to aim for range between 5.0 and 6.0. scores closest to 6.0 are optimal
    normalised_readability = (1.0 / (1.0 + abs(readability_score - 6.0)))
    return normalised_readability

def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        origICD10codes : dict,
        use_memory: bool,
        model: Model,
        letter: str
    ) -> Tuple[List[Dict[str, Any]], str]:
    
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)

    with open('alfworld_runs/base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"

    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs):
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

        if env_config["is_success"]:
            num_successes += 1

            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        #generate patient letter with and without Reflexion
        base_prompt = "Improve this patient-friendly medical report, to ensure that the readability scores a 6 on the Flesch-Kincaid Grade Level metric, and make sure the medical content of the letter matches that of the original medical report. The generated ICD-10 codes should be identical. List the most relevant ICD-10 codes from this medical report. Return the ICD-10 codes in JSON format, for example - ('I70.8': 'Atherosclerosis of other arteries') with the heading '### ICD-10 Codes'. \n\n{letter}"
        refined_patient_letter = model_instance.generate(origICD10codes, base_prompt, ob, letter) #version
        print("INPUT LETTER", letter)
        #print("REFLECTED", refined_patient_letter)

    return env_configs, refined_patient_letter

#choose the best version using reflexion model
def choose_best_version(versions: List[Tuple[str, float, float]], icd10codes: dict, original_report: str, icd10codesOrig: dict, model: Model = "gpt-4o") -> str:
    #llm has to give consistent and accurate outputs. ground truth needs to be accurate and identical for every run. prompt engineering.
    best_score = -1
    best_version = None

    for trial_idx, version_data in enumerate(versions):
        version = version_data['letter']
        readability = check_readability(version)
        accuracy = check_accuracy(icd10codes, icd10codesOrig)
        version_data['readability'] = readability
        version_data['accuracy'] = accuracy
        
        #initial weighted scores
        score = (readability * 0.3) + (accuracy * 0.7)
        print("ORIGINAL SCORE:", score)

        #reflection trials
        trial_log_path = f"trial_log_{trial_idx}.txt"
        world_log_path = f"world_log_{trial_idx}.txt"
        env_configs = [{'is_success': False, 'memory': [version]}]
        print("ABOUT TO RUN_TRIAL")
        env_configs, refined_letter = run_trial(trial_log_path, world_log_path, trial_idx, env_configs, icd10codesOrig, use_memory=True, model=model, letter=version)
        
        #calculate new scores for refined letter
        new_readability = check_readability(refined_letter)
        new_accuracy = check_accuracy(json.loads(refined_letter.split('```json')[1].split('```')[0]), icd10codesOrig)
        new_score = (new_readability * 0.3) + (new_accuracy * 0.7)
        print("NEW SCORE:", new_score)

        if new_score > best_score:
            best_score = new_score
            best_version = refined_letter
            
    return best_version

#generate and choose the best patient-friendly report
def generate_patient_friendly_report(radiology_report: str, model: Model = "gpt-4o") -> str:
    prompt = f"This is the original medical radiology report. Please turn this medical radiology report into a patient-friendly version:\n\n{radiology_report}. Please evaluate all generated patient-friendly letters and produce ICD-10 codes from the content of each letter. List the most relevant ICD-10 codes from this medical report. Return the ICD-10 codes in JSON format, for example - ('I70.8': 'Atherosclerosis of other arteries') with the heading '### ICD-10 Codes'. The patient-friendly letter should be at a 6th-grader's comprehension level (Flesch-Kincaid score between 5.0 - 6.0)" #Please Assign the variable 'accuracy' for each patient-friendly version to the proportion of ICD-10 codes in each patient-friendly version that exactly match the ICD-10 codes in the original medical report."
    versions = llm(prompt, model, num_versions=3)
    #print(versions)
    icd10dictionary = dict()
    for version in versions:
        #print("version example", version)
        #print("icd-10 codes", version.split("### ICD-10 Codes"))
        #icd10codes = version.split("### ICD-10 Codes")#[1]
        icd10dictionary = json.loads(version['letter'].split('```json')[1].split('```')[0])
        #print("icd10 dict", icd10dictionary)
    orig_icd10 = get_original_icd10_codes(radiology_report)
    print("ORIGINAL ICD10:", orig_icd10)
    best_version = choose_best_version(versions, icd10dictionary, radiology_report, orig_icd10, model)
    return best_version, icd10dictionary

#integration; pulling from SIIM FHIR Patient Repository
def get_FHIR_DiagnosticReport_Resource():
    response = requests.request("GET", FHIR_URL+"DiagnosticReport/a819497684894128", headers=FHIR_HEADERS).json()
    #test usage
    html_report = response["text"]["div"]
    filter = HTMLFilter()
    filter.feed(html_report)
    return filter.text 

#check accuracy - do icd10 codes from the original medical report match those generated from the content of the patient report?
def check_accuracy(icd10codes: dict, origCodes: dict):
    accuracy = 0
    totalCodes = len(origCodes)
    for code in origCodes:
        #print("CODE:", code)
        #isolate code and description, use colon as a delimiter
        if code in icd10codes and icd.is_valid_item(code): # checks if a code exists
            #generated_desc = icd10codes[code]
            #correct_desc = icd.get_description(code) # get the correct description from the ICD-10 catalogue
            #print("DESCRIPTION COMPARISON:", generated_desc, correct_desc)
            #if generated_desc == correct_desc: #compare llm output with description in ICD-10 catalogue 
            #comparing the description using pure string matching ended up producing too many false negatives by a difference of one word etc.
            accuracy += 1
    accuracy_score = accuracy / totalCodes
    #print("ORIGINAL REPORT CODES", origCodes)
    #print("PATIENT LETTER CODES", icd10codes)
    #print("ACCURACY SCORE:", accuracy_score)

    return accuracy_score

    #if accuracy_score >= 0.7:
    #    return accuracy_score #guardrails to ensure that accuracy is above a certain threshold. set to 0.8 
    #else:
        #restart llm process to generate patient-friendly letters OR return an error message? good chance that the accuracy will be low during a second pass too
        #(final_report, icd10codes) = generate_patient_friendly_report(radiology_report)
    #    return 0

#get ICD-10 Codes from the original report 
def get_original_icd10_codes(radiology_report: str):
    original_icd10_body = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": f"{radiology_report}\n\nList the most relevant ICD-10 codes from this medical report. Return the output in JSON format, for example - ('I70.8': 'Atherosclerosis of other arteries'). Do not return anything else and no do not wrap the JSON in markdown"},  # user prompt
                ],
                "temperature": 0 #want to keep these responses consistent and predictable. used for comparison
            }
    icd_10_original = requests.post(MD_AI_URL, json=original_icd10_body, headers=MD_AI_HEADERS).json()
    icd_10_original = icd_10_original["response"]["choices"][0]["message"]["content"]
    icd_10_original = eval(icd_10_original) #llm_icd_codes = {"1.2": 'PE', "1.3" : "XYZ"}
    return icd_10_original


def parse_patient_friendly_report(report: str, icd10codes: dict):
    letter_code = CodeableConcept()
    letter_coding = Coding()
    letter_coding.system = "http://loinc.org"
    letter_coding.code = "68609-7"
    letter_code.coding = list()
    letter_code.coding.append(letter_coding)

    deviceReferenceObject = ReferenceType()
    deviceReferenceObject.reference = "Device/md.ai"

    patientReferenceObject = ReferenceType()
    patientReferenceObject.reference = "Patient/siimirene"

    compositionLetterSection = {"title": "Patient Letter", "text": {"status":"generated", "div": report}}
    compositionCodesSection = {"title": "ICD-10 Codes", "text": {"status":"generated", "div": str(icd10codes)}}

    compositionObject = Composition(date=DateTime.now(), status="preliminary", title="Patient Letter", author=[{"reference":"Device/MDai"}], subject=[{"reference":"Patient/siimirene"}], type=letter_code.json(), section=[compositionLetterSection, compositionCodesSection])

    return compositionObject

def create_communication_object(composition: Composition):
    #print(composition.section[0].text.div)
    communicationObject = Communication(
        status="preparation",
        subject=Reference(reference=composition.subject[0].reference),
        recipient=[Reference(reference=composition.subject[0].reference)],
        sender=Reference(reference=composition.author[0].reference),
        payload=[{"contentReference": Reference(reference=composition.section[0].text.div)}]  # Corrected line
    )
    return communicationObject


def post_composition_resource(composition: Composition):
    response = requests.request("POST", FHIR_URL+"Composition", headers=FHIR_HEADERS, data=composition.json()).json()
    return response
def post_communication_resource(communication: Communication):
    response = requests.request("POST", FHIR_URL+"Communication", headers=FHIR_HEADERS, data=communication.json()).json()
    return response

##test radiology report
#radiology_report = get_FHIR_DiagnosticReport_Resource()

radiology_report = '''
History: [Patient presents with pelvic pain and irregular menstrual cycles.]

Last menstrual period: [Patient reports last menstrual period was 3 weeks ago.]

Technique: Ultrasound of the pelvis was performed utilizing [transabdominal and transvaginal approach] assessing gray scale appearance and color Doppler flow.

Comparison: [None]

Findings

Uterus:
Size: [8 x 4 x 5] cm
Orientation: [Anteverted]
Appearance: [Homogeneous myometrium.] [No discrete fibroids identified.]

Endometrium: [15] mm in thickness. [No mass or increased vascularity.]

Cervix: [Unremarkable.]

Right ovary:
Size: [3 x 2 x 2] cm
Appearance: [Normal]
Flow: [Normal]

Left ovary:
Size: [5 x 4 x 3] cm
Appearance: [Heterogeneous, hypoechoic mass suggestive of an ovarian cyst with possible hemorrhagic content.]
Flow: [Normal]

[No free pelvic fluid.]

Impression:

[1. Heterogeneous, hypoechoic mass in the left adnexa measuring approximately 5 cm, suggestive of an ovarian cyst with possible hemorrhagic content.
2. Thickened endometrial lining at 15 mm, raising concerns for endometrial hyperplasia.
3. No free fluid in the cul-de-sac.]
'''

print("ORIGINAL RAD REPORT", radiology_report)
readability_res = readability.getmeasures(radiology_report, lang='en')
print("READABILITY OF MEDICAL REPORT:", readability_res['readability grades']['Kincaid'])

#generate final report
(final_report, icd10codes) = generate_patient_friendly_report(radiology_report)
print("Patient-friendly report:\n", final_report)
compositionObject = parse_patient_friendly_report(final_report, icd10codes)
communicationObject = create_communication_object(compositionObject)
#uncomment below lines after debugging is completed
print(compositionObject.json())
print(communicationObject.json())
print(post_composition_resource(compositionObject))
print(post_communication_resource(communicationObject))