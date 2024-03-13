import os
from PyPDF2 import PdfWriter, PdfReader

def merge_pdfs_in_folder(folder_path, output_folder, output_filename):
    writer = PdfWriter()

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            filepath = os.path.join(folder_path, filename)
            reader = PdfReader(filepath)
            for page in reader.pages:
                writer.add_page(page)

    # Write the merged PDF to the output file in the specified output folder
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'wb') as output_file:
        writer.write(output_file)

# Example usage
folder_path = 'I:/elestator/GPT-/Data'  # Replace this with the path to your folder containing PDF files
output_folder = 'I:/elestator/GPT-/Data_update'  # Replace this with the path to your output folder
output_filename = 'merged_file.pdf'  # Output filename for the merged PDF

merge_pdfs_in_folder(folder_path, output_folder, output_filename)
