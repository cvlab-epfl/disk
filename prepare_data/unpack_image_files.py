import tarfile

def image_filter(member, path):
    member = tarfile.data_filter(member, path)
    if member is None:
        return None

    #if not member.name.endswith('images.txt'):
    if not member.name.endswith('cameras.txt'):
        return None

    print(f'Allowing {member.name}.')
    return member

with tarfile.open('/cvlabdata2/cvlab/datasets_tyszkiewicz/megadepth/MegaDepth_SfM_v1.tar.xz', mode='r:xz') as archive:
    archive.extractall(path='/cvlabdata1/cvlab/datasets_tyszkiew/megadepth', filter=image_filter)
