from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from io import BytesIO
import requests 
import re
import csv
import seaborn as sns
import matplotlib.pyplot as plt

# get character names from table
names = set()

NAMES_URL = 'https://www.reddit.com/r/Re_Zero/wiki/charbios/'

response = requests.get(NAMES_URL)

soup = BeautifulSoup(response.text, features='html.parser')

table = soup.find('table')

for row in table.find_all('tr'): 
    first_td = row.find('td')
    if first_td:
        name_parts = first_td.text.split(" ")
        names.add(name_parts[0])
        names.add(name_parts[-1])

print(names)

# list of arc 4 pdfs 
pdf_urls = ["https://witchculttranslation.com/wp-content/uploads/2018/11/a4-ph01-c001-c023-unrevised.pdf", 
"https://witchculttranslation.com/wp-content/uploads/2018/11/a4-ph02-c024-c051-unrevised.pdf",
"https://witchculttranslation.com/wp-content/uploads/2018/11/a4-ph03-c052-c079-unrevised.pdf",
"https://witchculttranslation.com/wp-content/uploads/2018/11/a4-ph04-c080-c097-unrevised.pdf",
"https://witchculttranslation.com/wp-content/uploads/2018/11/a4-ph05-c098-c117-unrevised.pdf",
"https://witchculttranslation.com/wp-content/uploads/2018/11/a4-ph06-c118-end-unrevised.pdf"]

# get features and labels 
full_text = []

for url in pdf_urls:

  response = requests.get(url, verify=False)

  reader = PdfReader(BytesIO(response.content))

  text_chunks = []

  for page in reader.pages: 
      text = page.extract_text()
      if text: 
        text_chunks.append(text)

  pdf_text = '\n'.join(text_chunks)

  full_text.append(pdf_text)

full_text = "".join(full_text)

print(full_text)

pattern = re.compile(rf"({'|'.join(names)}): “(.*?)”")

matches = re.findall(pattern, full_text)

print(matches)

print("Number of dialogues:", len(matches))

# save to file
with open('data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Text', 'Label'])
    writer.writerows([(text, label) for label, text in matches])

df = pd.read_csv('data.csv') 

# plot data 

label_counts = df['Label'].value_counts().reset_index()
label_counts.columns = ['Character', 'Count']
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.barplot(data=label_counts, x="Character", y="Count", palette="muted")
plt.title("Number of Lines per Character")
plt.xlabel("Character")
plt.ylabel("Number of Lines")
plt.tight_layout()
plt.show()
