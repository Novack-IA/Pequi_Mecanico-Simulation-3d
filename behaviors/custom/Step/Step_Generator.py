import math


class Step_Generator():
    GRAVITY = 9.81
    Z0 = 0.2
    
    def __init__(self, feet_y_dev, sample_time, max_ankle_z) -> None:
        self.feet_y_dev = feet_y_dev
        self.sample_time = sample_time
        self.state_is_left_active = False
        self.state_current_ts = 0
        self.switch = False # trocar as pernas
        self.external_progress = 0 # progresso não sobreposto
        self.max_ankle_z = max_ankle_z


    def get_target_positions(self, reset, ts_per_step, z_span, z_extension):
        '''
        Get target positions for each foot

        Returns
        -------
        target : `tuple`
            (Left leg y, Left leg z, Right leg y, Right leg z)
        '''

        assert type(ts_per_step)==int and ts_per_step > 0, "ts_per_step must be a positive integer!"

        #-------------------------- Avançar 1ts
        if reset:
            self.ts_per_step = ts_per_step        # duração do passo em passos de tempo
            self.swing_height = z_span
            self.max_leg_extension = z_extension  #distância máxima entre o tornozelo e o centro de ambas as articulações do quadril
            self.state_current_ts = 0
            self.state_is_left_active = False 
            self.switch = False
        elif self.switch:
            self.state_current_ts = 0
            self.state_is_left_active = not self.state_is_left_active # mudar de perna
            self.switch = False
        else:
            self.state_current_ts += 1

        #-------------------------- Calcular COM.y
        W = math.sqrt(self.Z0/self.GRAVITY)

        step_time = self.ts_per_step * self.sample_time
        time_delta = self.state_current_ts * self.sample_time
 
        y0 = self.feet_y_dev # valor inicial absoluto de y
        y_swing = y0 + y0 * (  math.sinh((step_time - time_delta)/W) + math.sinh(time_delta/W)  ) / math.sinh(-step_time/W)

        #-------------------------- Extensão máxima da tampa e altura de giro
        z0 = min(-self.max_leg_extension, self.max_ankle_z) # valor z inicial limitado
        zh = min(self.swing_height, self.max_ankle_z - z0) # altura de balanço tampada

        #-------------------------- Calcular balanço Z
        progress = self.state_current_ts / self.ts_per_step
        self.external_progress = self.state_current_ts / (self.ts_per_step-1)
        active_z_swing = zh * math.sin(math.pi * progress)

        #-------------------------- Aceitar novos parâmetros após a etapa final
        if self.state_current_ts + 1 >= self.ts_per_step:
            self.ts_per_step = ts_per_step        # duração do passo em passos de tempo
            self.swing_height = z_span
            self.max_leg_extension = z_extension  #distância máxima entre o tornozelo e o centro de ambas as articulações do quadril
            self.switch = True

        #-------------------------- Distinguir perna ativa
        if self.state_is_left_active:
            return y0+y_swing, active_z_swing+z0, -y0+y_swing, z0
        else:
            return y0-y_swing, z0, -y0-y_swing, active_z_swing+z0

