import sys
import os
import json
import requests
from typing import List, Dict, Any, Tuple
from utils import Model
import readability
import simple_icd_10 as icd
from html.parser import HTMLParser
import yaml
import importlib
from env_history import EnvironmentHistory


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import alfworld
import alfworld.alfworld.agents.environment
from alfworld_runs import *

from importlib import reload
from fhir.resources.communication import Communication
from fhir.resources.composition import Composition
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.fhirtypes import DateTime, ReferenceType
from fhir.resources.reference import Reference

class PatientLetterEnv:
    def __init__(self):
        self.sessions = {}

    def step(self, session, action):
        done = False
        observation_ = None
        if action == 'reset':
            self.sessions[session] = {'session': session, 'stage': 'init'}
            observation_ = "Session initialized."
        elif action.startswith('generate'):
            assert self.sessions[session]['stage'] == 'init'
            prompt = action[9:-1]
            letter = self.generate_letter(prompt)
            self.sessions[session] = {'session': session, 'stage': 'generated', 'letter': letter}
            observation_ = letter
        elif action.startswith('refine'):
            assert self.sessions[session]['stage'] == 'generated'
            feedback = action[7:-1]
            refined_letter = self.refine_letter(self.sessions[session]['letter'], feedback)
            self.sessions[session]['letter'] = refined_letter
            observation_ = refined_letter
            done = True
        else:
            assert False, "Invalid action"

        observation = observation_ if observation_ else "Invalid action"
        reward = 1.0 if done else 0.0
        return observation, reward, done

class Model:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, context: str) -> str:
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a Primary Care Physician, and your job is to generate a patient-friendly letter that is medically and factually accurate, in an appropriate tone and language, matching the reading level of an 8th grader if it was measured using the Flesch–Kincaid readability test"},  # system prompt
                {"role": "user", "content": f"{prompt}\n\n{context}"},  # user prompt
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


class HTMLFilter(HTMLParser):
    text = ""
    def handle_data(self, data):
        self.text += data

#loading prompt file
FOLDER = 'alfworld_runs/prompts'
PROMPT_FILE = 'alfworld_3prompts.json'

with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

# URL and headers for the md.ai endpoint
MD_AI_URL = 'https://staging.md.ai/api/openai/chat/completions'
MD_AI_HEADERS = {
    'Content-Type': 'application/json',
    'x-access-token': '233498d212978d8b28db2497889f4581'  # alter token to personal token
}
FHIR_URL = "https://hackathon.siim.org/fhir-r4/"
FHIR_HEADERS = {
    'apikey': '75ea6e1d-ee38-439a-bca7-d7876440c570',
    'Content-Type': 'application/json'
}

#prompting LLM 
def llm(prompt: str, model: Model, stop: List[str] = ["\n"], num_versions: int = 5) -> List[str]:
    versions = []

    for i in range(num_versions):
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a Primary Care Physician, and your job is to generate a patient-friendly letter that is medically and factually accurate, in an appropriate tone and language, matching the reading level of an 8th grader if it was measured using the Flesch–Kincaid readability test"},  # system prompt
                {"role": "user", "content": prompt},  # user prompt
            ],
            "temperature": i * 0.3  # play around with this. higher, more variance
        }
        response = requests.post(MD_AI_URL, json=body, headers=MD_AI_HEADERS)
        response_data = response.json()
        choices = response_data.get('response', {}).get('choices', [])
        if choices:
            text = choices[0].get('message', {}).get('content', '')
            if len(text.strip()) >= 5:
                versions.append(text)
    return versions

def check_readability(text: str) -> float:
    #Flesch-Kincaid readability score - score of 7.0-8.9 indicates 8th grade level
    readability_result = readability.getmeasures(text, lang='en')
    readability_score = readability_result['readability grades']['Kincaid']
    normalised_readability = (1.0 / (1.0 + abs(readability_score - 8.0)))
    return normalised_readability

#reflexion may work better on complex reports. Test this and evaluate metrics.
#try all 5 SIIM reports in both methods (reflexion vs not) and compare outputs
#then test on 5 reports generated by each of us (daily generations)
#check what datasets prev papers have used to validate

def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model: Model
    ) -> List[Dict[str, Any]]:
    env = PatientLetterEnv() #check if this needs to be changed

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs):
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

        print(f"using {name}")

        if env_config["is_success"]:
            num_successes += 1
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        base_prompt = "Improve this patient-friendly medical report, to ensure that the readability scores an 8 on the Flesch-Kincaid Grade Level metric, and make sure the medical content of the letter matches that of the original medical report. The generated ICD-10 codes should be identical."
        refined_patient_letter = model.generate(base_prompt, ob)

        try:
            final_env_history, is_success = alfworld_run(env, base_prompt, env_config["memory"] if use_memory else [], to_print=True, ob=ob, model=model_instance)
            if is_success:
                status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                env_configs[z]["is_success"] = True
                num_successes += 1
                num_additional_successes += 1
            else:
                status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

        except AssertionError:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}:\nAssertion Error\n\nSTATUS: FAIL\n\n#####\n')

        with open(world_log_path, 'a') as f:
            f.write(status_str + '\n')

    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    print("REFLEXION LETTER:", refined_patient_letter)
    return env_configs

def choose_best_version(versions: List[str], icd10codes: dict, original_report: str, icd10codesOrig: dict, model: Model = "gpt-4o") -> str:
    best_score = -1
    best_version = None

    for trial_idx in range(len(versions)):
        version = versions[trial_idx]
        readability = check_readability(version)
        accuracy = check_accuracy(icd10codes, icd10codesOrig)

        score = (readability * 0.3) + (accuracy * 0.7)

        if score > best_score:
            best_score = score
            best_version = version

        trial_log_path = f"trial_log_{trial_idx}.txt"
        world_log_path = f"world_log_{trial_idx}.txt"
        env_configs = [{'is_success': False, 'memory': [version]}]
        env_configs = run_trial(trial_log_path, world_log_path, trial_idx, env_configs, use_memory=True, model=model_instance)
        
        for env_config in env_configs:
            if env_config['is_success']:
                version = env_config['memory'][0]
                readability = check_readability(version)
                accuracy = check_accuracy(icd10codes, icd10codesOrig)
                score = (readability * 0.3) + (accuracy * 0.7)
                if score > best_score:
                    best_score = score
                    best_version = version

    return best_version

def generate_patient_friendly_report(radiology_report: str, model: Model = "gpt-4o") -> str:
    prompt = f"This is the original medical radiology report. Please turn this medical radiology report into a patient-friendly version:\n\n{radiology_report}. Please evaluate all generated patient-friendly letters and produce ICD-10 codes from the content of each letter. List the most relevant ICD-10 codes from this medical report. Return the ICD-10 codes in JSON format, for example - ('description':'Hypertension','code':'I10')."
    
    versions = llm(prompt, model)
    icd10codesOrig = {'original_icd_10_code': 'code1', 'original_icd_10_code_2': 'code2'}  # Modify based on actual data
    icd10codes = {'icd_10_code': 'code1', 'icd_10_code_2': 'code2'}  # Modify based on actual data

    best_version = choose_best_version(versions, icd10codes, radiology_report, icd10codesOrig, model=model)
    
    return best_version


#generate and choose the best patient-friendly report
def generate_patient_friendly_report(radiology_report: str, model: Model = "gpt-4o") -> str:
    prompt = f"This is the original medical radiology report. Please turn this medical radiology report into a patient-friendly version:\n\n{radiology_report}. Please evaluate all generated patient-friendly letters and produce ICD-10 codes from the content of each letter. List the most relevant ICD-10 codes from this medical report. Return the ICD-10 codes in JSON format, for example - ('I70.8': 'Atherosclerosis of other arteries') with the heading '### ICD-10 Codes'. The patient-friendly letter should be at an 8th-grader's comprehension level (Flesch-Kincaid score between 7.0 - 8.9)" #Please Assign the variable 'accuracy' for each patient-friendly version to the proportion of ICD-10 codes in each patient-friendly version that exactly match the ICD-10 codes in the original medical report."
    versions = llm(prompt, model, num_versions=1)
    #print(versions)
    icd10dictionary = dict()
    for version in versions:
        #print("version example", version)
        #print("icd-10 codes", version.split("### ICD-10 Codes"))
        #icd10codes = version.split("### ICD-10 Codes")#[1]
        icd10dictionary = json.loads(version.split('```json')[1].split('```')[0])
        #print("icd10 dict", icd10dictionary)
    orig_icd10 = get_original_icd10_codes(radiology_report)
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

def alfworld_run(env, base_prompt, memory: List[str], to_print=True, ob='', model = model_instance) -> Tuple[EnvironmentHistory, bool]:
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, [])
    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0
    while cur_step < 49:
        action = model.generate(env_history.__str__() + ">", stop=['\n']).strip()
        env_history.add("action", action)
        
        if action.startswith('generate:'):
            prompt = action[len('generate:'):].strip()
            observation = env.generate_letter(prompt)
        elif action.startswith('refine:'):
            feedback = action[len('refine:'):].strip()
            observation = env.refine_letter(env_history.actions[-1][1], feedback)
        else:
            observation = 'Invalid action!'
        
        env_history.add("observation", observation)
        if to_print:
            print(f'> {action}\n{observation}')
            sys.stdout.flush()
        if "done" in observation.lower():
            return env_history, True
        elif env_history.check_is_exhausted():
            return env_history, False
        cur_step += 1
    return env_history, False

#check accuracy - do icd10 codes from the original medical report match those generated from the content of the patient report?
def check_accuracy(icd10codes: dict, origCodes: dict):
    accuracy = 0
    totalCodes = len(origCodes)
    for code in origCodes:
        #print("CODE:", code)
        #isolate code and description, use colon as a delimiter
        if code in icd10codes and icd.is_valid_item(code): # checks if a code exists
            generated_desc = icd10codes[code]
            correct_desc = icd.get_description(code) # get the correct description from the ICD-10 catalogue
            
            if generated_desc == correct_desc: #compare llm output with description in ICD-10 catalogue 
                accuracy += 1
    accuracy_score = accuracy / totalCodes
    print("ACCURACY SCORE:", accuracy_score)
    return accuracy_score

    """
    if accuracy_score >= 0.7:
        return accuracy_score #guardrails to ensure that accuracy is above a certain threshold. set to 0.8 
    else:
        #restart llm process to generate patient-friendly letters OR return an error message? good chance that the accuracy will be low during a second pass too
        #(final_report, icd10codes) = generate_patient_friendly_report(radiology_report)
        return 0
    """

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
    icd_10_original = eval(icd_10_original)
    print("ORIGINAL CODES", icd_10_original) #llm_icd_codes = {"1.2": 'PE', "1.3" : "XYZ"}
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
radiology_report = get_FHIR_DiagnosticReport_Resource()
# '''
# CT Chest with Contrast
# Clinical History: Abnormal chest x-ray
# Comparison: Chest XR 1/1/00, Chest CT 10:04 AM 1/1/00
# Technique: Contiguous axial CT images of the chest were obtained after the administration of IV contrast material.
# Findings:
# The heart is normal in size. There is no pericardial effusion.
# The thoracic aorta is normal in size and calibur. There is a bovine configuration of the aortic arch, a normal variant. Mild atherosclerotic changes are present in the upper abdominal aorta.
# There is a 2.1 x 1.4 cm cavitary lesion in the left upper lobe (Image 28/116). A smaller cavitary lesion is seen in the right lower lobe measuring 9 mm (Image 40/116). Linear bandlike areas of scarring/atelectasis are present in the right upper lobe with associated bronchiectasis. The central airways are clear.
# There is no axillary, mediastinal, or hilar adenopathy. The thyroid gland is within normal limits.
# The visualized portions of the upper abdomen are unremarkable. No suspicious osseous lesion is seen. There is evidence of prior trauma/deformity of the right 5th rib.
# Impression:
#     Interval evolution of left upper lobe cavitary lesion in the left upper and new right lower lobe nodule. Given the rapid evolution of these cavitary lesions an infectious etiology is favored.
#     Unchanged bandlike areas of atelectasis/scarring and bronchiectasis in the right upper lobe likely related to prior infection/insult.
# '''

#generate final report
(final_report, icd10codes) = generate_patient_friendly_report(radiology_report)
print("Patient-friendly report:\n", final_report)

# final_report = '''
# Patient-friendly report:
#  ### Patient-Friendly Letter
# Dear [Patient's Name],
# I hope this letter finds you well. I wanted to share the results of your recent CT scan of the chest, which was done with a special dye to help us see things more clearly.
# Here’s what we found:
# 1. **Heart and Blood Vessels**: Your heart looks normal in size, and there is no fluid around it. The main blood vessel in your chest, called the thoracic aorta, is also normal. There are some mild changes in another part of this blood vessel in your upper abdomen, but nothing to worry about right now.
# 2. **Lungs**: We found a small cavity (a hollow space) in the upper part of your left lung, measuring about 2.1 x 1.4 cm. There is also a smaller cavity in the lower part of your right lung, about 9 mm in size. These cavities might be due to an infection, as they have appeared quite quickly.
# 3. **Scarring and Airways**: There are some areas of scarring and small airway changes in the upper part of your right lung. This is likely from a past infection or injury. The main airways in your lungs are clear.
# 4. **Lymph Nodes and Thyroid**: The lymph nodes in your chest and armpits are normal, and your thyroid gland is also normal.
# 5. **Other Areas**: The parts of your upper abdomen that we could see look normal. There is no sign of any suspicious bone problems, although we did notice an old injury or deformity in your right 5th rib.
# **What This Means**:
# - The cavities in your lungs are likely due to an infection, given how quickly they have appeared.
# - The scarring and airway changes in your right lung are probably from a past infection or injury and have not changed since your last scan.
# We will need to follow up to understand more about these findings and decide on the best course of action. Please make an appointment so we can discuss this in more detail and plan any further tests or treatments if needed.
# Take care and see you soon.
# Best regards,
# [Your Doctor's Name]
# '''


# icd10codes = {
#     "J98.4": "Other disorders of lung",
#     "I70.0": "Atherosclerosis of aorta",
#     "J84.10": "Pulmonary fibrosis, unspecified",
#     "J47.9": "Bronchiectasis, uncomplicated",
#     "S22.41XA": "Fracture of one rib, right side, initial encounter for closed fracture"
# }

compositionObject = parse_patient_friendly_report(final_report, icd10codes)
communicationObject = create_communication_object(compositionObject)
#uncomment below lines after debugging is completed
# print(compositionObject.json())
# print(communicationObject.json())
# print(post_composition_resource(compositionObject))
# print(post_communication_resource(communicationObject))

# Example usage
radiology_report = "Original radiology report text goes here."
patient_friendly_report = generate_patient_friendly_report(radiology_report)
print(patient_friendly_report)
