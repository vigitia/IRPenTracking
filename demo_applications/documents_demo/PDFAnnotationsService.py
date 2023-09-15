import os
import time

from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2.generic import DictionaryObject, NumberObject, FloatObject, NameObject, TextStringObject, ArrayObject

DEBUG_MODE = False

FILENAME_INPUT = 'berlin.pdf'
FILENAME_OUTPUT = 'berlin_out.pdf'
FULL_PATH_INPUT = os.path.join(os.getcwd(), '../../assets', FILENAME_INPUT)
FULL_PATH_OUTPUT = os.path.join(os.getcwd(), '../../assets', FILENAME_OUTPUT)


class PDFAnnotationsService:

    last_modification_timestamp = 0

    highlights = []  # All text highlighted
    notes = []  # Virtual notes in the document
    freehand_lines = []  # Lines drawn into the pdf

    def __init__(self):
        print('[PdfAnnotationsExtractionService]: Ready')
        self.document_path = FULL_PATH_INPUT
        self.writer = PdfFileWriter()

        self.reset_document_annotations_from_output_file()

        self.__fetch_document_data()

    # Get all again from the document
    def __fetch_document_data(self):
        self.input_pdf = PdfFileReader(open(self.document_path, "rb"))
        self.page_width = self.input_pdf.getPage(0).mediaBox[2]
        self.page_height = self.input_pdf.getPage(0).mediaBox[3]
        self.num_pages = self.input_pdf.getNumPages()

        if DEBUG_MODE:
            print('[PdfAnnotationsExtractionService]: PDF page dimensions ({}, {}). Num pages: {}'.
                  format(self.page_width, self.page_height, self.num_pages))

    # Check if a change has appeared in the document since we last checked
    def __has_something_changed(self):
        last_modification_timestamp = self.__get_timestamp_last_modification()
        if last_modification_timestamp != self.last_modification_timestamp:
            self.last_modification_timestamp = last_modification_timestamp
            print('------------------------DOCUMENT HAS CHANGED!----------------------------')
            return True
        return False

    # Extract from the pdf when the last change has happened
    def __get_timestamp_last_modification(self):
        # https://thispointer.com/python-get-last-modification-date-time-of-a-file-os-stat-os-path-getmtime/
        last_modification_timestamp = os.path.getmtime(FULL_PATH_INPUT)
        # Convert seconds since epoch to readable timestamp
        # modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modification_timestamp))
        # print("Last Modified Time : ", modificationTime, last_modification_timestamp)
        return last_modification_timestamp

    # Get all annotations (highlights, notes, etc. from the pdf)
    def get_annotations(self):

        # Read in all info about annotations again if the modification timestamp of the pdf indicates recent changes.
        # Otherwise, return previously stored info about existing annotations
        has_something_changed = self.__has_something_changed()
        if has_something_changed:
            self.highlights = []  # All text highlighted
            self.notes = []  # Virtual notes in the document
            self.freehand_lines = []  # Lines drawn into the pdf
            self.__fetch_document_data()

            # TODO: For all pages
            # for i in range(self.num_pages):
            current_page = self.input_pdf.getPage(0)
            try:
                # Get all annotations from the PDF
                for annotations in current_page['/Annots']:
                    annot_dict = annotations.getObject()

                    # Select all elements of type "Highlight"
                    if annot_dict['/Subtype'] == '/Highlight':
                        if DEBUG_MODE:
                            print('Highlight:', annot_dict)
                        highlight = {
                            # 'rect': self.sort_points(self.translate_points(annot_dict['/Rect'])),
                            'quad_points': self.__sort_points(self.__translate_points(annot_dict['/QuadPoints'])),
                            'color': annot_dict['/C'],
                            'annotator': annot_dict['/T'],
                            'timestamp': annot_dict['/M']
                        }
                        # print(highlight)
                        self.highlights.append(highlight)

                    # Select all elements of type "FreeText" -> Inline notes
                    elif annot_dict['/Subtype'] == '/FreeText':
                        if DEBUG_MODE:
                            print('Inline note:', annot_dict)
                        note = {
                            'rect': self.__translate_points(annot_dict['/Rect']),
                            'color': annot_dict['/C'],
                            'annotator': annot_dict['/T'],
                            'content': annot_dict['/Contents']
                        }
                        self.notes.append(note)

                    # Select all elements of type "Freehand Line"
                    elif annot_dict['/Subtype'] == '/Ink':
                        if DEBUG_MODE:
                            print('Freehand line:', annot_dict)
                        freehand_line = {
                            'rect': self.__translate_points(annot_dict['/Rect']),
                            'color': annot_dict['/C'],
                            'annotator': annot_dict['/T'],
                            'points': self.__translate_points(annot_dict['/InkList'][0])
                        }
                        self.freehand_lines.append(freehand_line)
            except Exception as e:
                print(e)

        # print('Extracted Highlights:', self.highlights)
        # print('Extracted Notes:', self.notes)
        # print('Freehand Lines:', self.freehand_lines)

        # self.__fetch_document_data()

        return self.highlights, self.notes, self.freehand_lines, has_something_changed

    # Flipping points along horizontal axis to match the default coordinate system
    # -> (0,0) in top-left corner (vs. (0,0) in bottom-left corner)
    def __translate_points(self, points_list):
        for i, point in enumerate(points_list):
            if i & 1 != 0:
                points_list[i] = FloatObject(self.page_height - point)

        return points_list

    # Sort a list of points into a list of corners in this order:
    # top_left -> bottom_left -> bottom_right -> top_right
    @staticmethod
    def __sort_points(points_list):
        num_rects = int(len(points_list) / 8)
        sorted_points = []

        for i in range(num_rects):
            rect_points = points_list[i*8:(i+1)*8]

            top_left = rect_points[0:2]
            top_right = rect_points[2:4]
            bottom_left = rect_points[4:6]
            bottom_right = rect_points[6:8]

            sorted_points.append([top_left[0], top_left[1], bottom_left[0], bottom_left[1], bottom_right[0],
                                  bottom_right[1], top_right[0], top_right[1]])

        return sorted_points

    # Get the bounding-rect for a line. It is needed for adding a new line to the pdf
    def __get_rect(self, line):
        hor_helper = line[:][::2]
        ver_helper = line[1:][::2]
        return [min(hor_helper)-1, min(ver_helper)-1, max(hor_helper)+1, max(ver_helper)+1]

    def add_lines_to_pdf(self, lines):
        for line in lines:
            self.add_annotation_to_pdf(self.create_line_data_structure(line))

    def create_line_data_structure(self, line):

        # TODO: Replace these hardcoded values
        color = [1, 1, 0]
        border = [0, 0, 2]
        author = 'PEN'
        contents = ''
        m = 'D:20210907121746'
        new_ink = DictionaryObject()

        rect = self.__get_rect(line)

        new_ink.update({
            NameObject("/F"): NumberObject(4),
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Ink"),

            NameObject("/T"): TextStringObject(author),
            NameObject("/Contents"): TextStringObject(contents),
            # NameObject("/CA"): 1,
            # NameObject("/P"): parent?,
            # NameObject("/M"): TextStringObject(m),

            NameObject("/Border"): ArrayObject([FloatObject(b) for b in border]),
            NameObject("/C"): ArrayObject([FloatObject(c) for c in color]),
            NameObject("/Rect"): ArrayObject([FloatObject(p) for p in rect]),
            # NameObject("/InkList"): ArrayObject([self.__translate_points(line['points'])]),
            NameObject("/InkList"): ArrayObject([ArrayObject([FloatObject(p) for p in line])])
        })
        return new_ink

    # Create a new highlight that can be added to the document
    # (x1, y1) is the bottom left corner, (x2, y2) the top left corner (Corresponding to the default PDF coordinate
    # system)
    # Function basted on: https://gist.github.com/agentcooper/4c55133f5d95866acdee5017cd318558
    def create_highlight_data_structure(self, highlight):
        x1, y1, x2, y2 = highlight['rect']

        # TODO: Replace these hardcoded values
        color = [1, 0, 0]
        author = 'VIGITIA'
        contents = 'Hello'

        new_highlight = DictionaryObject()

        new_highlight.update({
            NameObject("/F"): NumberObject(4),
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Highlight"),

            NameObject("/T"): TextStringObject(author),
            NameObject("/Contents"): TextStringObject(contents),

            NameObject("/C"): ArrayObject([FloatObject(c) for c in color]),
            NameObject("/Rect"): ArrayObject([
                FloatObject(x1),
                FloatObject(y1),
                FloatObject(x2),
                FloatObject(y2)
            ]),
            NameObject("/QuadPoints"): ArrayObject([
                FloatObject(x1),
                FloatObject(y2),
                FloatObject(x2),
                FloatObject(y2),
                FloatObject(x1),
                FloatObject(y1),
                FloatObject(x2),
                FloatObject(y1)
            ]),
        })
        return new_highlight

    # Add a new highlight to the specified page and save the document
    # Function based on https://gist.github.com/agentcooper/4c55133f5d95866acdee5017cd318558
    def add_annotation_to_pdf(self, annotation):
        # print('Adding new annotation:', annotation)
        page = self.input_pdf.getPage(0)

        annotation_ref = self.writer._add_object(annotation)

        if "/Annots" in page:
            page[NameObject("/Annots")].append(annotation_ref)
        else:
            page[NameObject("/Annots")] = ArrayObject([annotation_ref])

    def write_changes_to_file(self):
        # print('write new annotations to PDF')
        file_writer = PdfFileWriter()
        file_writer.appendPagesFromReader(self.input_pdf)

        with open(FULL_PATH_OUTPUT, "wb") as out_file:
            file_writer.write(out_file)

        self.__fetch_document_data()

    def reset_document_annotations_from_output_file(self):
        print('RESET PDF')
        file_writer = PdfFileWriter()
        file_writer.appendPagesFromReader(PdfFileReader(open(self.document_path, "rb")))
        file_writer.removeLinks()  # Delete all Annotations from PDF

        with open(FULL_PATH_OUTPUT, "wb") as out_file:
            file_writer.write(out_file)


if __name__ == '__main__':

    # Test
    pdf_annotations_service = PDFAnnotationsService()
    highlights, notes, freehand_lines = pdf_annotations_service.get_annotations()

    for highlight in highlights:
        print('New Highlight:')
        for rectangle in highlight['quad_points']:
            print('Highlight points:', rectangle)
            # TODO: Sort the rects into the following format: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    # line needs to be a list of points, e.g: [x1, y1, x2, y2, x3, y3]
    test_lines = [[100, 0, 150, 100, 350, 550, 10, 300]]
    pdf_annotations_service.add_lines_to_pdf(test_lines)
    pdf_annotations_service.write_changes_to_file()
