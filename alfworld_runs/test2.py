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
from env_history import EnvironmentHistory
from fhir.resources.communication import Communication
from fhir.resources.composition import Composition
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.fhirtypes import DateTime, ReferenceType
from fhir.resources.reference import Reference

class EnvironmentHistory:
    def __init__(self):
        self.steps = []

    def add_step(self, observation: str, action: str, reward: float):
        self.steps.append({'observation': observation, 'action': action, 'reward': reward})

    def __str__(self):
        return "\n".join([f"Step {i+1}: {step['action']} -> {step['observation']} (Reward: {step['reward']})" for i, step in enumerate(self.steps)])


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
            letter = llm(prompt, model_instance, 3)
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

    def reset(self):
        session_id = 'default'
        self.sessions[session_id] = {'session': session_id, 'stage': 'init'}
        observation = "Session initialized."
        info = {'extra.gamefile': ['default_game_file']}
        return observation, info


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

    def refine_letter(self, prompt: str, context: str) -> str:
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "Evaluate this patient-friendly letter. Do the ICD-10 codes match the content? Are the descriptions correct? Ensure that the language is as close as possible to US 6th grade level."},  # system prompt
                {"role": "user", "content": f"{prompt}\n\n{context}"},  # user prompt
            ],
            "temperature": 0.5
        }
        response = requests.post(MD_AI_URL, json=body, headers=MD_AI_HEADERS)
        response_data = response.json()
        choices = response_data.get('response', {}).get('choices', [])
        if choices:
            text = choices[0].get('message', {}).get('content', '')
            return text
        return ""


class HTMLFilter(HTMLParser):
    text = ""

    def handle_data(self, data):
        self.text += data


# Loading prompt file
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


# Prompting LLM
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
    readability_result = readability.getmeasures(text, lang='en')
    readability_score = readability_result['readability grades']['Kincaid']
    normalised_readability = (1.0 / (1.0 + abs(readability_score - 8.0)))
    return normalised_readability


def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model: Model
) -> List[Dict[str, Any]]:
    env = PatientLetterEnv()

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


def alfworld_run(env: PatientLetterEnv, base_prompt: str, memory: List[str], to_print: bool, ob: str, model: Model) -> Tuple[EnvironmentHistory, bool]:
    history = EnvironmentHistory()
    done = False
    step_counter = 0

    while not done:
        if step_counter == 0:
            action = f"generate({base_prompt})"
        else:
            action = f"refine({memory[-1]})"

        observation, reward, done = env.step('default', action)
        step_counter += 1
        history.add_step(observation, action, reward)

        if to_print:
            print(f"Step {step_counter}:")
            print(f"Action: {action}")
            print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print("\n")

        memory.append(observation)

    is_success = reward > 0  # Assuming that a reward > 0 indicates success
    return history, is_success


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

        num_trial_idx = 1
        env_configs = [{
            "memory": [],
            "is_success": False
        }]
        use_memory = True
        env_configs = run_trial(trial_log_path, world_log_path, num_trial_idx, env_configs, use_memory, model)

    return best_version

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

    if accuracy_score >= 0.7:
        return accuracy_score #guardrails to ensure that accuracy is above a certain threshold. set to 0.8 
    else:
        #restart llm process to generate patient-friendly letters OR return an error message? good chance that the accuracy will be low during a second pass too
        #(final_report, icd10codes) = generate_patient_friendly_report(radiology_report)
        return 0

def process_fhir(resource: Communication, refined_letter: str) -> Communication:
    text = f"""
    <div xmlns="http://www.w3.org/1999/xhtml">
    <p>{refined_letter}</p>
    </div>
    """
    resource.payload = [{"contentString": text}]
    resource.status = "completed"
    resource.text.div = text
    resource.text.status = "generated"
    return resource


# Loading ICD-10 codes for medical terms
icd_codes = {
    "hypertension": "I10",
    "diabetes": "E11",
    "hyperlipidemia": "E78.5"
}

# Sample prompt
prompt = "The patient is a 45-year-old male with a history of hypertension and diabetes. He has been on medication for the past 10 years and has been stable. Recent tests show elevated cholesterol levels. The patient is advised to make dietary changes and increase physical activity. The generated ICD-10 codes should be identical."
original_report = prompt

# Step 1: Generate multiple versions of the patient letter using the LLM
model_instance = Model("gpt-4o")
versions = llm(prompt, model_instance, num_versions=5)

# Step 2: Choose the best version
best_version = choose_best_version(versions, icd_codes, original_report, icd_codes, model=model_instance)

# Step 3: Create a FHIR Communication resource
communication_resource = Communication.construct(
    status="in-progress",
    text={"status": "generated", "div": f"<div xmlns='http://www.w3.org/1999/xhtml'><p>{best_version}</p></div>"},
    payload=[{"contentString": best_version}]
)

# Step 4: Refine the FHIR Communication resource with the final letter
final_fhir_resource = process_fhir(communication_resource, best_version)

# Step 5: Print or use the FHIR resource as needed
print(json.dumps(final_fhir_resource.dict(), indent=2))
