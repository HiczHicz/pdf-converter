from src.utils import DataExplorer

helper = DataExplorer()

folders = helper.get_all_dirs()
files = helper.get_files_in_dir(folders[1])
pogrupowane = helper.group_files_by_extension(files)
pliki_pdf = helper.filter_by_extension(pogrupowane, ".pdf")
for folder in folders:
    files = helper.get_files_in_dir(folders[1])
    for file in files:
        pogrupowane = helper.group_files_by_extension(files)
        pliki_pdf = helper.filter_by_extension(pogrupowane, ".pdf")
        for pdf_file in pliki_pdf:
            # Tutaj kontynuacja - docelowo pewnie do opakowania w funkcje
            continue
