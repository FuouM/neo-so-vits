import os


tt = "F:\\1 New Home\\Archive 8\\supertest\\so-vits-svc\\raw\\test.wav"

input_files_path = [tt]
input_files = [os.path.basename(file).split('.')[0] for file in input_files_path]
print(input_files)