"""Scientific regressions: complete horizons, real time, failure semantics."""
from __future__ import annotations
import csv,json
import numpy as np
import pytest
from hidden_attractors.basins.classification import class_label,is_target_class
from hidden_attractors.workflows import basin_runner as runner

class ProbeSystem:
    rhs=staticmethod(lambda t,x: np.array([1.,0.,0.]))
    def equilibrium_points(self):return {'E0':np.zeros(3)}

def payload(**overrides):
    data={'h':.1,'T':1.,'burn':.2,'reference':np.ones((9,3))*50.,'eq_tol':.1,'policy':{}}
    data.update(overrides)
    return (0,0,np.zeros(3),ProbeSystem(),'rk4',1.,data['h'],data['T'],data['burn'],
            data['reference'],[np.zeros(3)],data['eq_tol'],120.,'centroid_distance',.1,
            'full',None,False,data['policy'])

def classify(monkeypatch,times,states,status='ok',**kwargs):
    seen={}
    def integrate(**params):seen.update(params);return np.array(times),np.array(states),status
    monkeypatch.setattr(runner,'integrate',integrate)
    return runner._classify_point_worker(payload(**kwargs)),seen

def test_early_capture_then_escape_cannot_be_equilibrium(monkeypatch):
    times=np.linspace(0,1,11);states=np.zeros((11,3));states[7:,0]=[1.,3.,6.,121.]
    result,seen=classify(monkeypatch,times,states,'diverged')
    assert result[2]==3
    assert result[3]['reason']=='observed_finite_norm_threshold'
    assert seen['early_stop_config']['enabled'] is False
    assert seen['early_stop_config']['equilibrium_enabled'] is False

def test_an_endpoint_near_equilibrium_is_not_terminal_capture(monkeypatch):
    states=np.ones((11,3));states[-1]=0
    result,_=classify(monkeypatch,np.linspace(0,1,11),states)
    assert result[2]==2
    assert result[3]['tail_span']>1

@pytest.mark.parametrize('status',['converged_equilibrium_early','ok'])
def test_short_solution_never_replaces_burn_or_completes_horizon(monkeypatch,status):
    result,_=classify(monkeypatch,[0.,.05,.1],np.zeros((3,3)),status,burn=.2)
    assert result[2]==4
    assert result[3]['tail_samples']==0

@pytest.mark.parametrize('status',['nonfinite_solution','diverged'])
def test_nonfinite_is_numerical_failure_not_norm_event(monkeypatch,status):
    states=np.zeros((11,3));states[-1,0]=np.nan
    result,_=classify(monkeypatch,np.linspace(0,1,11),states,status)
    assert result[2]==4
    assert result[3]['reason']=='nonfinite_solution'

def test_growth_heuristic_below_explicit_threshold_is_incomplete(monkeypatch):
    result,_=classify(monkeypatch,[0.,.1,.2],np.ones((3,3))*10,'diverged_early')
    assert result[2]==4
    assert result[3]['max_norm']<120

def test_tail_selection_uses_actual_time_instead_of_h_index(monkeypatch):
    times=[0.,.1,.6,.8,1.];states=np.ones((5,3));states[2:]=0
    result,_=classify(monkeypatch,times,states,burn=.5,h=.001)
    assert result[2]==0
    assert result[3]['tail_start']==.6
    assert result[3]['tail_samples']==3

def test_explicit_terminal_window_is_recorded(monkeypatch):
    states=np.ones((11,3));states[-3:]=0
    result,_=classify(monkeypatch,np.linspace(0,1,11),states,policy={'terminal_window':.2})
    assert result[2]==0
    assert result[3]['tail_start']==pytest.approx(.8)

def test_contact_is_operational_and_has_metric(monkeypatch):
    states=np.ones((11,3))*3
    result,_=classify(monkeypatch,np.linspace(0,1,11),states,reference=np.ones((5,3))*3)
    assert result[2]==1
    assert result[3]['reference_contact']['value']==0
    assert result[3]['reference_contact']['meaning']=='finite_reference_contact_only'

def test_reference_horizon_required_and_early_stop_disabled(tmp_path,monkeypatch):
    seen={}
    monkeypatch.setattr(runner,'get_system',lambda _:ProbeSystem())
    def integrate(**kw):seen.update(kw);return np.array([0.,.1,.2]),np.zeros((3,3)),'ok'
    monkeypatch.setattr(runner,'integrate',integrate)
    with pytest.raises(ValueError,match='incomplete_horizon'):
        runner.run_basin_workflow({'run_basin_slices':True,'system_id':'stub','q':1.,'integrator':'rk4','h':.1,'output_dir':str(tmp_path),'early_stop':{'equilibrium_enabled':True},'final_simulation':{'t_final':1.,'t_burn':.2},'basin':{'t_final':1.,'t_burn':.2}})
    assert seen['early_stop_config']['equilibrium_enabled'] is False
    assert seen['early_stop_config']['enabled'] is False
    assert not (tmp_path/'reference_attractor.csv').exists()

def test_policy_aware_target_ids_preserve_legacy_reader():
    assert class_label(1)=='target_positive'
    assert is_target_class(1) and is_target_class(2,classifier_policy='legacy_v1')
    assert class_label(1,classifier_policy='terminal_v2')=='bounded_positive_mean'
    assert not is_target_class(1,classifier_policy='terminal_v2')
    assert not is_target_class(2,classifier_policy='terminal_v2')
    with pytest.raises(ValueError):is_target_class(1,classifier_policy='unknown')

def test_both_contact_metrics_and_invalid_metric():
    cloud=np.array([[1.,0.,0.],[2.,1.,0.],[3.,2.,0.]])
    assert runner._target_contact_measurement(cloud,cloud,'nn_percentile',0.)['value']==0
    assert runner._evaluate_target_match(cloud,cloud,'bbox_overlap')
    with pytest.raises(ValueError):runner._evaluate_target_match(cloud,cloud,'typo')

def test_disabling_early_stops_never_mutates_user_config():
    original={'enabled':True,'equilibrium':{'enabled':True,'tol':.1}}
    effective=runner._basin_early_stop_config(original)
    assert original['enabled'] and original['equilibrium']['enabled']
    assert effective['equilibrium']['tol']==.1
    assert effective['enabled'] is False

def test_local_slice_default_plane_passes_through_its_equilibrium(tmp_path,monkeypatch):
    class Shifted(ProbeSystem):
        def equilibrium_points(self):return {'E+':np.array([1.,2.,3.])}
    seen=[]
    monkeypatch.setattr(runner,'get_system',lambda _:Shifted())
    monkeypatch.setattr(runner,'classify_equilibrium_stability',lambda *args:{'stable':True})
    def integrate(**kw):
        x=np.array(kw['x0']);seen.append(x);return np.linspace(0,1,11),np.tile(x,(11,1)),'ok'
    monkeypatch.setattr(runner,'integrate',integrate)
    cfg={'run_basin_slices':True,'system_id':'stub','q':1.,'integrator':'rk4','h':.1,'output_dir':str(tmp_path),'initial_conditions':{'ref':[1.,2.,3.]},'plot_enabled':False,'final_simulation':{'t_final':1.,'t_burn':.2},'basin':{'enabled':True,'planes':['xy'],'grid_n':2,'t_final':1.,'t_burn':.2,'h':.1,'around_equilibria':True}}
    summary=runner.run_basin_workflow(cfg)
    assert all(x[2]==3 for x in seen)
    assert summary['results'][0]['fixed_coordinate']=={'z':3.}

def test_plot_extent_and_labels_match_finite_sampled_domain(tmp_path,monkeypatch):
    from hidden_attractors.plotting import basin,export
    captured={}
    def capture(fig,*args):
        ax=fig.axes[0];captured.update(xlim=ax.get_xlim(),ylim=ax.get_ylim(),title=ax.get_title(),legend=[x.get_text() for x in ax.get_legend().get_texts()])
    monkeypatch.setattr(export,'intercept_and_export_path',capture)
    basin.plot_basin_slice_file('xy',np.array([-1.,0.,1.]),np.array([-.1,0.,.1]),np.zeros((3,3)),
                               'global','test',tmp_path)
    assert captured['xlim']==(-1.,1.) and captured['ylim']==(-.1,.1)
    assert 'Finite destination map' in captured['title']
    assert 'Reference contact (finite)' in captured['legend']
    assert not any('Attractor' in label for label in captured['legend'])
