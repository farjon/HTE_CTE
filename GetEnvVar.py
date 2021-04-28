import os

def GetEnvVar(varName):
    computerInfo = os.environ
    env = {}
    env['userName'] = os.getlogin()
    env['computerName'] = computerInfo['USERDOMAIN_ROAMINGPROFILE']

    if env['computerName'] == 'SRV-AHARON3' and env['userName'] == 'mstryoda':  # Guy on server
        env['storagePath'] = 'D:\\Guy\\drive\\StoragePath\\'

    elif env['computerName'] == 'COMP10259':  # Guy's new PC
        env['storagePath'] = 'C:\\Users\\owner\\Google Drive\\StoragePath\\'

    elif env['computerName'] == 'SRV-AHARON1':  # Aharon CPU server
        env['storagePath'] = 'C:\\Users\\aharon\\Google Drive\\StoragePath\\'

    elif env['computerName'] == 'DESKTOP-4QDODMV':  # Aharon's laptop
        env['storagePath'] = 'D:/GoogleDrive/StoragePath/'

    elif env['computerName'] == 'DESKTOP-3BTL5ES': #Omri_Dahan laptop
        env['storagePath'] = 'D:\\Google Post Drive\\StoragePath\\'


    env['DatasetsPath'] = env['storagePath'] + 'Datasets\\'
    env['ModelsPath'] = env['storagePath'] + 'Models\\'
    env['ExternalCodePath'] = env['storagePath'] + 'ExternalCode\\'
    env['ExpResultsPath'] = env['storagePath'] + 'ExpResults\\'

    if varName in env.keys():
        return env[varName]
    else:
        print("%s does not exists" %(varName))
        return None

if __name__ == "__main__":
    print('Calling GetEnvVar.....')
    a = GetEnvVar('storagePath')
    print('Storage path is ' + a)