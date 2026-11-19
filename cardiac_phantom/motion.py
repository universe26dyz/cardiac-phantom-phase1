"""Analytic material deformation and world-coordinate respiratory translation."""
from dataclasses import dataclass, field
import numpy as np
from . import Phantom

@dataclass
class Motion:
    phantom: Phantom = field(default_factory=Phantom)
    cardiac_period: float = 1.0
    respiratory_period: float = 4.0
    cardiac_amplitude: float = 0.5
    longitudinal_amplitude: float = 0.10
    respiratory_ap_mm: float = 4.0
    respiratory_si_mm: float = 8.0
    cardiac_phase: float = 0.0
    respiratory_phase: float = 0.0
    sigma_mm: float = 18.0
    def __post_init__(self):
        if self.cardiac_period <= 0 or self.respiratory_period <= 0 or self.sigma_mm <= 0 or not 0 <= self.cardiac_amplitude < .95 or not 0 <= self.longitudinal_amplitude < .95:
            raise ValueError('invalid period, scale, or amplitude')
    def factors(self,t,cardiac=True,respiratory=True):
        c=(np.cos(self.cardiac_phase)-np.cos(2*np.pi*t/self.cardiac_period+self.cardiac_phase))/2 if cardiac else 0.
        b=np.sin(2*np.pi*t/self.respiratory_period+self.respiratory_phase)-np.sin(self.respiratory_phase) if respiratory else 0.
        return c, np.array([0.,self.respiratory_ap_mm*b,self.respiratory_si_mm*b])
    def canonical_to_world(self,xyz,t,cardiac=True,respiratory=True):
        p=np.asarray(xyz,dtype=float); c,b=self.factors(t,cardiac,respiratory); r2=np.sum(p[...,:2]**2,axis=-1)
        scale=1-self.cardiac_amplitude*c*np.exp(-r2/(2*self.sigma_mm**2))
        q=p.copy(); q[...,:2]*=scale[...,None]; q[...,2]*=1-self.longitudinal_amplitude*c
        return self.phantom.local_to_world(q)+b
    def world_to_canonical(self,xyz,t,cardiac=True,respiratory=True):
        c,b=self.factors(t,cardiac,respiratory); q=self.phantom.world_to_local(np.asarray(xyz,dtype=float)-b)
        target=np.linalg.norm(q[...,:2],axis=-1); lo=target.copy(); hi=target/(1-self.cardiac_amplitude*abs(c))
        for _ in range(55):
            r=(lo+hi)/2; value=r*(1-self.cardiac_amplitude*c*np.exp(-r*r/(2*self.sigma_mm**2)))
            lo=np.where(value<target,r,lo); hi=np.where(value>=target,r,hi)
        r=(lo+hi)/2; scale=np.divide(r,target,out=np.ones_like(r),where=target>0)
        p=q.copy(); p[...,:2]*=scale[...,None]; p[...,2]/=1-self.longitudinal_amplitude*c
        if np.any(np.abs(r*(1-self.cardiac_amplitude*c*np.exp(-r*r/(2*self.sigma_mm**2)))-target)>1e-8): raise RuntimeError('cardiac inverse failed tolerance')
        return p
    def query(self,xyz,t,cardiac=True,respiratory=True):
        return self.phantom.query(self.phantom.local_to_world(self.world_to_canonical(xyz,t,cardiac,respiratory)))
    def dvf(self,canonical,t,cardiac=True,respiratory=True):
        return self.canonical_to_world(canonical,t,cardiac,respiratory)-self.phantom.local_to_world(canonical)
