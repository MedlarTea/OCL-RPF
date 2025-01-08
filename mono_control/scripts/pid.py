



class PID_controller:
	def __init__(self, Kp, Ki, Kd, deadband, u_min, u_max, e_int_min, e_int_max, dt = 0.1):
		# dt is the integration step for the simulation
		self.e_cur	= 0.0
		self.e_old 	= 0.0
		self.e_int 	= 0.0
		self.e_der 	= 0.0
		self.ref 	= 0.0
		self.u 		= 0.0

		self.Kp 	= Kp
		self.Kd 	= Kd
		self.Ki 	= Ki

		self.dt 		= dt
		self.deadband 	= deadband
		self.e_int_max	= e_int_max
		self.e_int_min 	= e_int_min
		self.u_max		= u_max
		self.u_min 		= u_min

	def calc_output(self, x , dt):
		self.dt = dt


		self.e_cur = self.ref - x

		# Deadband
		if self.e_cur <= self.deadband / 2 and self.e_cur >= - self.deadband / 2: self.e_cur = 0

		self.e_der = (self.e_cur - self.e_old) / self.dt
		self.e_int = self.e_int + self.e_cur * self.dt

		# Limits for integral action
		if self.e_int > self.e_int_max: self.e_int = self.e_int_max
		if self.e_int < self.e_int_min: self.e_int = self.e_int_min

		self.u = self.Kp * self.e_cur + self.Kd * self.e_der + self.Ki * self.e_int

		# Limit controller action

		if self.u > self.u_max: self.u = self.u_max
		if self.u < self.u_min: self.u = self.u_min


		self.e_old = self.e_cur

		return self.u