import os

def GetEnvVar(varName):
    computerInfo = os.environ
    env = {}
    env['userName'] = os.getlogin()
    env['computerName'] = computerInfo['USERDOMAIN_ROAMINGPROFILE']

    if env['computerName'] == 'L-131W12':  # Guy's computer
        env['storagePath'] = 'D:\\Drive\\StoragePath\\'

    elif env['computerName'] == 'SRV-AHARON3' and env['userName'] == 'mstryoda':  # Guy on server
        env['storagePath'] = 'D:\\Guy\\drive\\StoragePath\\'

    elif env['computerName'] == 'COMP10259':  # Guy's new PC
        env['storagePath'] = 'C:\\Users\\owner\\Google Drive\\StoragePath\\'

    elif env['computerName'] == 'L-131W11':   # Omer's computer
        env['storagePath'] = 'G:\\My Drive\\StoragePath\\'  # 'G:\\My Drive\\StoragePath\\'
        env['Server_phenomics'] = '\\\\SRV-AHARON2\\phenomics'

    elif env['computerName'] == 'L-131W14':   # Alon's computer
        env['storagePath'] = 'D:\\google_drive\\StoragePath\\' #'G:\\My Drive\\StoragePath\\'
        # env['Server_phenomics'] = '\\\\SRV-AHARON2\\phenomics'

    elif env['computerName'] == 'AHARON1':   # Faina's computer
        env['storagePath'] = 'D:/Faina/StoragePath/'

    elif env['computerName'] == 'AHARON2':  # Yael's computer
        env['storagePath'] = 'G:\\My Drive\\StoragePath\\'

    elif computerInfo['computerName'] == 'LB117-4-GPU':   # Adar's computer
        env['storagePath'] = 'D:/Adar/StoragePath/'

    elif (computerInfo['computerName'] == 'ISE-GUY-SM1') or (env['computerName'] == 'ISE-GUY-SM1'):
        env['storagePath'] = 'D:/Adar/StoragePath/StoragePath/'

    elif env['computerName'] == 'DESKTOP-4QDODMV':  # Aharon's laptop
        env['storagePath'] = 'D:/GoogleDrive/StoragePath/'

    elif env['computerName'] == 'SRV-AHARON3' and env['userName'] == 'KyloRen':  # Omer on server
        env['storagePath'] = 'D:\\Omer_StoragePath\\StoragePath\\'
        # env['Server_phenomics'] = '\\\\SRV-AHARON2\\phenomics'

    elif env['computerName'] == 'SRV-AHARON3' and env['userName'] == 'Hansolo': #Atalia on server
        env['storagePath'] = 'D:\\Atalia\\Google Drive\\StoragePath\\'

    elif env['computerName'] == 'SRV-AHARON3' and env['userName'] == 'Skywalker': #Yael on server
        # env['storagePath'] = 'D:\\Faina_StoragePath\\StoragePath\\' #Faina on server
        env['storagePath'] = 'D:\\Yael\\drive\\StoragePath\\'
    elif env['computerName'] == 'SRV-AHARON3' and env['userName'] == 'catcherray': #Alon on server
        env['storagePath'] = 'D:\\Alon\\drive\\StoragePath\\'

    elif env['computerName'] == 'DESKTOP-662SBLD': #Alon on Itay computer
        env['storagePath'] = 'C:\\google_drive\\StoragePath\\'
        env['Server_phenomics'] = '\\\\SRV-AHARON2\\phenomics'

    elif env['computerName'] == 'LAPTOP-UR87BO4L': #Faina laptop
        env['storagePath'] = 'C:\\Users\\borde\\Google Drive\\StoragePath\\'

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