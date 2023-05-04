f = open("data.txt", encoding='utf-8')
file_lines = f.readlines()

from csv import writer
question_py = []
dps = []
dp = None
for line in file_lines:
  if line[0] == "#":
    if dp:
      dp['solution'] = ''.join(dp['solution'])
      dps.append(dp)
    dp = {"question": None, "solution": []}
    valu = dp['question'] = line[1:]
    # print(valu)
    question_py.append(valu)
    #     # cs.close()
  else:
    dp["solution"].append(line)

# print(dps['question'])
# print(question_py)


with open('questions.csv','a', encoding='utf-8') as cs:
    writer_object = writer(cs)
    writer_object.writerow(question_py)