## Agentic Workflows in Patient-Friendly Letter Generation

The application of Large Language Models (LLMs) in healthcare is expanding rapidly, with one potential use case being the translation of formal medical reports into patient-legible equivalents. Currently, LLM outputs often need to be edited and evaluated by a human to ensure both factual accuracy and comprehensibility, and this is true for the above use case. We aim to eliminate this step by proposing an agentic workflow using Reflexion, which uses iterative self-reflection to correct outputs from an LLM. This pipeline was tested and compared to zero-shot prompting on 16 randomized radiology reports. In our multi-agent approach, reports had an accuracy rate of 94.94%, when looking at verification of ICD-10 codes, compared to zero-shot prompted reports, which had an accuracy rate of 68.23%. Of the final reflected reports, 81.25% did not need to be corrected for accuracy or readability, compared to just 25% of zero-shot prompted patient reports that did not require any modifications. These results indicate that our approach is a feasible method for clinical findings to be communicated to patients in a quick, efficient and coherent manner whilst also retaining medical accuracy. 

### Running the Script

From the main `Multi-Agent-Patient-Letter-Generation` directory, navigate to the `alfworld_runs` folder where the script is located. Then, run the `patient-letter-generation.py` file by using the following command:

```sh
python alfworld_runs/patient-letter-generation.py
```

This command will start the multi-agent pipeline and generate the patient-friendly reports based on the given inputs. Remember to replace the OpenAI __[API-KEY]__ value and the sample medical report with your own parameters.