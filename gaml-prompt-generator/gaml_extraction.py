import glob
import re

# get all nested .gaml files in the "GAML/library" directory
gaml_files = glob.glob("/home/phanh/Downloads/finetuneGAMA/code/gaml_prompt_generator/GAML/library/**/*.gaml", recursive=True)

def clean_code_line(line):
    #line = line.replace('r {4,}', '')
    line = re.sub(r'\n {4,}\t* *', '', line)
    line = line.replace(';\n', ';')
    #print(line)
    return line
    
def extract_single_line(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        #print(text)
    single_line_regex = r'\n {4,}.*;'
    code_lines = re.findall(single_line_regex, text)

    # remove line contain words like: [write, assert, {, }, draw]
    stopword = ['write', 'assert', '{', '}', 'draw', 'transition', 'Organism', 'temp']
    code_lines = [line for line in code_lines if not any(word in line for word in stopword)]
    
    # clean code lines
    code_lines = [clean_code_line(line) for line in code_lines]
    #print(code_lines)

    return code_lines

total_code_lines = []
for file in gaml_files:
    total_code_lines.extend(extract_single_line(file))
# save to txt file
with open('library_gaml_code_lines.txt', 'w') as file:
    file.write('\n'.join(total_code_lines))

line = '	     width <- (rnd(100)/100)*(rnd(100)/100)*(rnd(100)/100)*50+10;'
line = re.sub(r'\t *', '', line)
print(line)