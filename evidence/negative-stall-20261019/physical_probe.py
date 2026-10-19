"""Bounded isolated final-state actuator diagnostic, never a learned policy."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import mujoco
ROOT=Path('/home/robot/.pi05-handson');sys.path.insert(0,str(ROOT/'scripts'))
from simulation_lab import LiberoTask,TARGET,state_digest
from generate_tracking_teacher import restore_controller
from libero.libero.envs import OffScreenRenderEnv
SESSION=ROOT/'evaluations/interactive_lab_retrain_review_1792383683641025226'
OUTPUT=ROOT/'.local/negative-stall-20261019/physical-probe.json'
assert not OUTPUT.exists(), 'Refusing overwrite'
task=LiberoTask()
report={'status':'running','hypothesis':'Can the saved failure end states move under small directed diagnostic control, or are they mechanically immovable?','policy_calls':0,'training_data_generated':False,'max_total_probe_control_steps':120,'actual_total_probe_control_steps':0,'setup_control_steps':0,'task_metadata':task.metadata,'branches':[],'source_code_sha256':{f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in ['simulation_lab.py','generate_tracking_teacher.py']},'limits':['This is diagnostic external control, not learned recovery or a demonstration added to training.','Integration state restores exactly; hidden OSC/controller state is refreshed consistently, not serialized exactly.','Contact forces below are instantaneous MuJoCo solver estimates after forward, not historical forces from GUI.','Only two final states and30steps per branch; mobility here does not exclude earlier collisions or identify model/data cause.','B uses constant norm.15 toward actual plate-body center from bowl center, zero rotation and closed gripper. No initial lift or C branch.']}
def save():OUTPUT.write_text(json.dumps(report,indent=2))
def snapshot(env,obs):
 model,data=env.sim.model,env.sim.data
 eef=np.asarray(obs['robot0_eef_pos']);bowl=np.asarray(obs[f'{TARGET}_pos']);plateid=env.env.obj_body_id['plate_1'];plate=np.asarray(data.body_xpos[plateid]).copy()
 contacts=[]
 for i,c in enumerate(data.contact[:data.ncon]):
  ids=[int(c.geom1),int(c.geom2)];names=[model.geom_id2name(j) for j in ids];bodies=[model.body_id2name(int(model.geom_bodyid[j])) for j in ids]
  force=np.zeros(6);mujoco.mj_contactForce(model._model,data._data,i,force)
  contacts.append({'geom_names':names,'body_names':bodies,'dist_m':float(c.dist),'force_contact_frame_6d':force.tolist(),'normal_force_n':float(force[0]),'position':np.asarray(c.pos).tolist(),'involves_target':any(TARGET in str(x) for x in names+bodies),'involves_robot':any('robot' in str(x) or 'gripper' in str(x) for x in names+bodies)})
 robot=env.env.robots[0];q=np.asarray(data.qpos[robot._ref_joint_pos_indexes]);r=np.asarray(model.jnt_range[robot._ref_joint_indexes]);margins=np.minimum(q-r[:,0],r[:,1]-q)
 return {'sim_time':float(data.time),'eef_xyz':eef.tolist(),'bowl_xyz':bowl.tolist(),'plate_xyz':plate.tolist(),'eef_plate_distance_m':float(np.linalg.norm(eef-plate)),'bowl_plate_distance_m':float(np.linalg.norm(bowl-plate)),'bowl_plate_xy_distance_m':float(np.linalg.norm((bowl-plate)[:2])),'finger_qpos':np.asarray(obs['robot0_gripper_qpos']).tolist(),'on':bool(env.check_success()),'arm_joints':[{'name':name,'qpos':float(v),'lower_rad':float(lim[0]),'upper_rad':float(lim[1]),'nearest_limit_margin_rad':float(m)} for name,v,lim,m in zip(robot.robot_joints,q,r,margins)],'min_joint_limit_margin_rad':float(margins.min()),'contacts':contacts}
save()
for number in [8,9]:
 source=SESSION/f'scene-{number:03}_init-024';state_path=source/'final_state.npy';state=np.load(state_path,allow_pickle=False);result=json.loads((source/'result.json').read_text());assert state_digest(state)==result['final_state_sha256'];assert result['task_metadata']==task.metadata
 rows=list(csv.DictReader((source/'actions.csv').open()));original=np.array([[float(r[f'action_{j}']) for j in range(7)] for r in rows[-30:]]);assert original.shape==(30,7)
 for arm in ['A_original_last30','B_constant_toward_plate']:
  env=OffScreenRenderEnv(bddl_file_name=str(task.bddl),camera_heights=256,camera_widths=256,ignore_done=True)
  try:
   env.seed(42);env.reset();obs=env.regenerate_obs_from_state(state);assert np.array_equal(env.get_sim_state(),state)
   controller=restore_controller(env);initial=snapshot(env,obs)
   delta=np.array(initial['plate_xyz'])-np.array(initial['bowl_xyz']);direction=.15*delta/np.linalg.norm(delta);constant=np.r_[direction,np.zeros(3),1.];assert np.linalg.norm(constant[:3])<=.15000000001
   entry={'scene':number,'branch':arm,'state_source':str(state_path),'state_file_sha256':hashlib.sha256(state_path.read_bytes()).hexdigest(),'integration_state_exact_restore':True,'controller_initialization':controller,'initial':initial,'constant_action_B':constant.tolist() if arm.startswith('B_') else None,'actions_csv_sha256':hashlib.sha256((source/'actions.csv').read_bytes()).hexdigest(),'steps':[]}
   for index in range(30):
    action=original[index] if arm.startswith('A_') else constant
    env.step(action.tolist());obs=env.regenerate_obs_from_state(env.get_sim_state());entry['steps'].append({'control_step':index+1,'action':action.tolist(),**snapshot(env,obs)});report['actual_total_probe_control_steps']+=1;assert report['actual_total_probe_control_steps']<=120
   last=entry['steps'][-1];entry['summary']={'eef_displacement_m':float(np.linalg.norm(np.array(last['eef_xyz'])-initial['eef_xyz'])),'bowl_displacement_m':float(np.linalg.norm(np.array(last['bowl_xyz'])-initial['bowl_xyz'])),'eef_plate_distance_change_m':last['eef_plate_distance_m']-initial['eef_plate_distance_m'],'bowl_plate_distance_change_m':last['bowl_plate_distance_m']-initial['bowl_plate_distance_m'],'bowl_plate_xy_distance_change_m':last['bowl_plate_xy_distance_m']-initial['bowl_plate_xy_distance_m'],'min_arm_limit_margin_over_branch_rad':min(x['min_joint_limit_margin_rad'] for x in [initial]+entry['steps']),'target_contact_body_pairs':sorted({tuple(c['body_names']) for x in [initial]+entry['steps'] for c in x['contacts'] if c['involves_target']})}
   report['branches'].append(entry);save();print(json.dumps({'scene':number,'branch':arm,'initial_eef':initial['eef_xyz'],'initial_bowl':initial['bowl_xyz'],'plate':initial['plate_xyz'],'initial_min_joint_margin_rad':initial['min_joint_limit_margin_rad'],'summary':entry['summary']}),flush=True)
  finally:env.close()
report['status']='complete';assert report['actual_total_probe_control_steps']==120
for number in [8,9]:
 source=SESSION/f'scene-{number:03}_init-024';assert state_digest(np.load(source/'final_state.npy'))==json.loads((source/'result.json').read_text())['final_state_sha256']
report['source_final_states_unchanged']=True;save()
