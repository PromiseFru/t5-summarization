import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
device = torch.device('cpu')

text = """Appendicitis is inflammation of the appendix.[2] Symptoms commonly include right lower abdominal pain, nausea, vomiting, and decreased appetite.[2] However, approximately 40% of people do not have these typical symptoms.[2] Severe complications of a ruptured appendix include widespread, painful inflammation of the inner lining of the abdominal wall and sepsis.[3]

Appendicitis is caused by a blockage of the hollow portion of the appendix.[10] This is most commonly due to a calcified "stone" made of feces.[6] Inflamed lymphoid tissue from a viral infection, parasites, gallstone, or tumors may also cause the blockage.[6] This blockage leads to increased pressures in the appendix, decreased blood flow to the tissues of the appendix, and bacterial growth inside the appendix causing inflammation.[6][11] The combination of inflammation, reduced blood flow to the appendix and distention of the appendix causes tissue injury and tissue death.[12] If this process is left untreated, the appendix may burst, releasing bacteria into the abdominal cavity, leading to increased complications.[12][13]
The diagnosis of appendicitis is largely based on the person's signs and symptoms.[11] In cases where the diagnosis is unclear, close observation, medical imaging, and laboratory tests can be helpful.[4] The two most common imaging tests used are an ultrasound and computed tomography (CT scan).[4] CT scan has been shown to be more accurate than ultrasound in detecting acute appendicitis.[14] However, ultrasound may be preferred as the first imaging test in children and pregnant women because of the risks associated with radiation exposure from CT scans.[4]

The standard treatment for acute appendicitis is surgical removal of the appendix.[6][11] This may be done by an open incision in the abdomen (laparotomy) or through a few smaller incisions with the help of cameras (laparoscopy). Surgery decreases the risk of side effects or death associated with rupture of the appendix.[3] Antibiotics may be equally effective in certain cases of non-ruptured appendicitis. [15][7] It is one of the most common and significant causes of abdominal pain that comes on quickly. In 2015 about 11.6 million cases of appendicitis occurred which resulted in about 50,100 deaths.[8][9] In the United States, appendicitis is one of the most common cause of sudden abdominal pain requiring surgery.[2] Each year in the United States, more than 300,000 people with appendicitis have their appendix surgically removed.[16] Reginald Fitz is credited with being the first person to describe the condition in 1886.[17]"""

preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=5,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)