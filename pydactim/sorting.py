import os
import pydicom
import shutil
from pydactim.anonymization import anonymize_dicom

def sort_dicom(dicom_dir, output_dir="", anonymize=False):
    """ Sort a Dicom dir by exam, patient, session and then sequence

    Parameters
    ----------
    dicom_dir : str
        Path of the directory that contains all the Dicom files to be stored

    output_dir : str
        Directory in which the Dicom files will be sorted. If blank, the 'dicom_dir' will be used as 'output_dir'

    anonymize : bool
        Whether anonymize the Dicom files or not

    """
    if output_dir == "":
        output_dir = dicom_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dicoms_path = [file_path for file_path in os.listdir(dicom_dir) if file_path.endswith(".dcm")]
    
    for dicom_path in dicoms_path:
        dicom_path = os.path.join(dicom_dir, dicom_path)
        if anonymize: anonymize_dicom(dicom_path, level=3)

        ds = pydicom.dcmread(dicom_path)
        if "." in str(ds.PatientID):
            examen_patient_id = str(ds.PatientName)
        else:
            examen_patient_id = str(ds.PatientName).upper() + "^"+  str(ds.PatientID)

        examen_sequence_series_number = str(ds.SeriesNumber)
        examen_sequence_name = examen_sequence_series_number + " " + str(ds.SeriesDescription).upper()
        examen_study_description = str(ds.StudyDescription).upper()
        examen_date = str(ds.StudyDate)

        # on va créer le dossier protocol dans sorted_data
        protocol_dir = os.path.join(output_dir, examen_study_description)
        if not os.path.exists(protocol_dir):
            os.makedirs(protocol_dir)

        # on defini le path patient
        patient_dir = os.path.join(protocol_dir, examen_patient_id)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)

        # on defini le path temps d'acquisition
        time_dir = os.path.join(patient_dir, examen_date)
        if not os.path.exists(time_dir):
            os.makedirs(time_dir)

        # on defini le path sequence
        sequence_dir = os.path.join(time_dir, examen_sequence_name)
        if not os.path.exists(sequence_dir):
            os.makedirs(sequence_dir)
            print("New path created :", sequence_dir)

        shutil.copy(dicom_path, sequence_dir)
        
        
