import os
import json
import pydicom
import time
from pydicom.uid import generate_uid

FILL_NUMBER = 3

def createAnonymisationTable(json_path):
    """ Create a anonymization table (json).

    Parameters
    ----------
    json_path : str
        The path of the JSON table file

    """
    with open(json_path, 'x') as f:
        data = {"anonymisation" : []}
        json.dump(data, f, indent=4)

def write(json_path, sub, patient_id):
    """ Write new patient in the anonymization table.

    Parameters
    ----------
    json_path : str
        The path of the JSON table file

    sub : str
        The key (unique id) of the new patient in bids format (ex: sub-001)

    patient_id : str
        The value (IPP) of the new patient (ex: firstname^lastname^IPP)
    """
    with open(json_path) as f:
        data = json.load(f)

    anonymisationDict = dict()
    anonymisationDict[sub] = patient_id
    data["anonymisation"].append(anonymisationDict)
        
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
   
def addOrGetSub(json_path, patient_id):
    """ Create the anonymization table if it is not existing or get the sub (unique id) of the patient.

    Parameters
    ----------
    json_path : str
        The path of the JSON table file

    patient_id : str
        The value (IPP) of the new patient (ex: firstname^lastname^IPP)

    Returns:
        str : the new or already existing sub (unique id) for the patient
    """
    if not os.path.exists(json_path):
        createAnonymisationTable(json_path)
        new_sub = "sub-" + str(1).zfill(FILL_NUMBER)
        write(json_path, new_sub, patient_id)
        return new_sub
    else:
        return getSubFromPatientID(json_path, patient_id)     

def getSubFromPatientID(json_path, patient_id):
    """ Look for a specific patient IPP in the anonymization table. If the patient is already in the table, it returns the unique id found in bids format of the patient.

    Parameters
    ----------
    json_path : str
        The path of the JSON table file

    patient_id : str
        The value (IPP) of the new patient (ex: firstname^lastname^IPP)

    Returns:
        str : the new or already existing sub (unique id) for the patient
    """
    with open(json_path) as f:
        data = json.load(f)
  
    liste = data["anonymisation"]
    for couple in liste:
        for sub in couple:
            if patient_id == couple[sub]:
                return sub

    for sub in liste[-1]:
        new_sub = "sub-" + str(int(sub[-3:]) + 1).zfill(FILL_NUMBER)
        write(json_path, new_sub, patient_id)
        return new_sub
    
def anonymize_dicom(dcm_path, level=1):
    """ Anonymize a Dicom file 

    Parameters
    ----------
    dcm_path : str
        The path of the Dicom file to anonymize

    level : int
        The level of anonymization (must be between 1 and 4).
        Level 1 : Patient information are removed
        Level 2 : Patient and date information are removed
        Level 3 : Patient, date and time information are removed
        Level 4 : Patient, date and time and uuid anonymized

    """
    print(f"INFO - Starting anonymization for\n\t{dcm_path :} with level {level}")
    ds = pydicom.dcmread(dcm_path, force=True)
    
    if level <= 1:
        ds.PatientName = 'Ano'
        ds.PatientID = 'Ano'
        ds.PatientBirthDate = '19000101'
        ds.PatientSex = 'Ano'
        ds.PatientAge = '00Y'
        ds.PatientSize = 0
        ds.PatientWeight = 0
    if level <= 2:
        ds.StudyDate = time.strftime("%Y%m%d")
        ds.SeriesDate = time.strftime("%Y%m%d")
        ds.AcquisitionDate = time.strftime("%Y%m%d")
        ds.ContentDate = time.strftime("%Y%m%d")
    if level <= 3:
        ds.StudyTime = time.strftime("%H%M%S")
        ds.SeriesTime = time.strftime("%H%M%S")
        ds.AcquisitionTime = time.strftime("%H%M%S")
        ds.ContentTime = time.strftime("%H%M%S")
    if level <= 4:
        uid = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = uid
        ds.SOPInstanceUID = uid
        
    ds.save_as(dcm_path)