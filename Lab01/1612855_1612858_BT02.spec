# -*- mode: python -*-

block_cipher = None


a = Analysis(['1612855_1612858_BT02.py'],
             pathex=['E:\\Visual Studio\\WorkSpace\\HCMUS\\[HK6][TGMT]\\CV_OpeNPlus\\Lab01'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='1612855_1612858_BT02',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
