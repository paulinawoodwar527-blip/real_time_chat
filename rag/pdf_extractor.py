from pypdf import PdfReader


def pdf_extractor(pdf_path):
    result = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)

            if reader.pages:
                for page in reader.pages:
                    text_content = page.extract_text()
                    result += text_content
            else:
                print("The PDF contains no pages.")
        
        return result
    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     print(pdf_extractor("C:\\Document1.pdf"))
    