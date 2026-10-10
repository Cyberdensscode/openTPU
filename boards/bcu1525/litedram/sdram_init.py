dfii_control_sel     = 0x01
dfii_control_cke     = 0x02
dfii_control_odt     = 0x04
dfii_control_reset_n = 0x08

dfii_command_cs     = 0x01
dfii_command_we     = 0x02
dfii_command_cas    = 0x04
dfii_command_ras    = 0x08
dfii_command_wrdata = 0x10
dfii_command_rddata = 0x20

ddrx_mr1 = 0x301

init_sequence = [
    ("Release reset", 0, 0, dfii_control_odt|dfii_control_reset_n, 50000),
    ("Release reset", 11256, 15, dfii_control_odt|dfii_control_reset_n, 50000),
    ("Bring CKE high", 0, 0, dfii_control_cke|dfii_control_odt|dfii_control_reset_n, 10000),
    ("Bring CKE high", 11256, 15, dfii_control_cke|dfii_control_odt|dfii_control_reset_n, 10000),
    ("Reset RCD", 96, 7, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 50000),
    ("Load RCD F0RC0F", 244, 7, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 100),
    ("Load RCD F0RC03", 53, 7, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 100),
    ("Load RCD F0RC04", 69, 7, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 100),
    ("Load RCD F0RC05", 85, 7, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 100),
    ("Load RCD F0RC0D", 212, 7, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 100),
    ("Load RCD F0RC0A", 160, 7, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 100),
    ("Load RCD F0RC3X", 786, 7, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 100),
    ("Load Mode Register 3", 0, 3, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 3", 11256, 12, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 6", 0, 6, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 6", 11256, 9, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 5", 1024, 5, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 5", 12280, 10, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 4", 0, 4, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 4", 11256, 11, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 2, CWL=9", 512, 2, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 2, CWL=9", 10744, 13, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 1", 769, 1, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 1", 10489, 14, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 0, CL=11, BL=8", 784, 0, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 200),
    ("Load Mode Register 0, CL=11, BL=8", 10472, 15, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 200),
    ("ZQ Calibration", 1024, 0, dfii_command_we|dfii_command_cs, 200),
    ("ZQ Calibration", 12280, 15, dfii_command_we|dfii_command_cs, 200),
]

phy = {'sys_hz': 200000000.0,
 'nphases': 4,
 'rdphase': 1,
 'wrphase': 3,
 'databits': 72,
 'dfi_databits': 144,
 'modules': 9,
 'delays': 512,
 'bitslips': 8,
 'cl': 11,
 'cwl': 9,
 'read_latency': 8,
 'write_latency': 2,
 'rdimm': True,
 'phy': 'uspddrphy',
 'memtype': 'DDR4',
 'part': 'xcvu9p-fsgd2104-2l-e',
 'idelay_ref_hz': 400000000.0}
