from fpdf import FPDF
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'AutoML EDA & Model Report', ln=True, align='C')
        self.ln(10)

    def add_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def add_image(self, image_path, w=160):
        self.image(image_path, w=w)
        self.ln(10)

    def add_text(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, text)
        self.ln()

def generate_pdf_from_charts(chart_paths, output_path="outputs/eda_report.pdf"):
    pdf = PDFReport()
    pdf.add_page()

    if isinstance(chart_paths, list):
        pdf.add_title("EDA Visualizations")
        for chart in chart_paths:
            if os.path.exists(chart):
                pdf.add_image(chart)
    else:
        pdf.add_title("Model Evaluation Plot")
        pdf.add_image(chart_paths)

    pdf.output(output_path)
    return output_path