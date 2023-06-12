import pickle
import pandas as pd
import xml.etree.ElementTree as ET

from flask import Flask, request, make_response, render_template
from beam_brep_3d import build_beam
from double_hat_solver import get_features_from_json

from OCC.Display.WebGl import x3dom_renderer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer


Interface_Static_SetCVal("write.step.schema", "AP242")

with open('load_ann_model.bin', 'rb') as load_model_file:
    load_transformer, load_model = pickle.load(load_model_file)

with open('deflection_ann_model.bin', 'rb') as deflection_model_file:
    deflection_transformer, deflection_model = pickle.load(deflection_model_file)


app = Flask(__name__)

# Just for testing purpose, we are going to read the L, A, R, t
# values from `input_dh_73.json` file
L, A, R, t = get_features_from_json('input_dh_73.json')

material_map = {'Usibor 2000': 0, 'Usibor 1500': 1, 'CP 900': 2, 'DP 1000': 3}


def make_dataframe(L, A, R, t, material):
    datadict = {}
    for i in range(8):
        datadict[f'L{i+1}'] = L[i]
        datadict[f'A{i+1}'] = A[i]
        datadict[f'R{i+1}'] = R[i]
    datadict['L9'] = L[8]
    datadict['t'] = t
    datadict['L'] = L[9]
    datadict['material'] = material
    return pd.DataFrame([datadict])


def predict_load_deflection(x_pred):
    """
    Given beam parameters calculate the load and deflections.
    """
    x_load = load_transformer.transform(x_pred)
    load_pred = load_model.predict(x_load)
    x_defl = deflection_transformer.transform(x_pred)
    defl_pred = deflection_model.predict(x_defl)
    predictions = f"""
    <Resultants>
    <Load>{round(load_pred[0])} N</Load>
    <Deflection>{round(defl_pred[0])} mm</Deflection>
    </Resultants>
    """ 
    return predictions


@app.route("/")
def root():
    return render_template("./index.html")


@app.route('/update_beam_material', methods=['POST'])
def update_beam_material():
    """
    Read the L, A, R, t, and material data sent from Javascript as a JSON,
    update the load and deflection values.
    """
    # Fetch the modified geometry parameters from the request
    data = request.json
    thickness = data.pop('t0')
    extrude_length = data.pop('L0')
    data['t'] = thickness
    data['L'] = extrude_length
    x_pred = pd.DataFrame([data])
    predictions = predict_load_deflection(x_pred)
    response = make_response(predictions)
    response.headers['Content-Type'] = 'text/xml'
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/update_beam_x3d', methods=['POST'])
def update_beam_x3d():
    """
    Read the L, A, R, t data sent from Javascript as a JSON,
    calculate the new beam geometry, prepare a new beam BREP,
    convert the brep to X3D and return the <triangleset> node
    of the X3D file.
    """
    # Fetch the modified geometry parameters from the request
    global L, A, R, t
    data = request.json
    material = data.pop('material')
    tochange = list(data.keys())[0];
    index = int(tochange[1])
    if tochange[0] == 'L':
        if index == 0:
            L[-1] = float(list(data.values())[0]);
        else:
            L[index - 1] = float(list(data.values())[0]);
    elif tochange[0] == 'A':
        A[index - 1] = float(list(data.values())[0]);
    elif tochange[0] == 'R':
        R[index - 1] = float(list(data.values())[0]);
    else:
        t = float(list(data.values())[0]);
    # Build the beam BREP model
    beam = build_beam(L, A, R, t)
    # Convert brep to x3dom format
    exporter = x3dom_renderer.X3DExporter(
        beam,
        vertex_shader=None,
        fragment_shader=None,
        export_edges=False,
        color=(0.65, 0.65, 0.7),
        specular_color=(0.2, 0.2, 0.2),
        shininess=0.9,
        transparency=0.0,
        line_color=(0, 0.0, 0.0),
        line_width=2.0,
        mesh_quality=1.0
    )
    exporter.compute()
    x3dstring = exporter.to_x3dfile_string(shape_id=0)
    x_pred = make_dataframe(L, A, R, t, material)
    predictions = predict_load_deflection(x_pred)
    predict_tag = ET.fromstring(predictions)
    x3d_tag = ET.fromstring(x3dstring)
    x3d_tag.append(predict_tag)
    new_tree_string = ET.tostring(x3d_tag)
    response = make_response(new_tree_string)
    response.headers['Content-Type'] = 'text/xml'
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/export_beam_brep', methods=['POST'])
def export_to_step():
    """
    Export STEP file
    """
    # Fetch the modified geometry parameters from the request
    global L, A, R, t
    data = request.json
    tochange = list(data.keys())[0];
    index = int(tochange[1])
    if tochange[0] == 'L':
        if index == 0:
            L[-1] = float(list(data.values())[0]);
        else:
            L[index - 1] = float(list(data.values())[0]);
    elif tochange[0] == 'A':
        A[index - 1] = float(list(data.values())[0]);
    elif tochange[0] == 'R':
        R[index - 1] = float(list(data.values())[0]);
    else:
        t = float(list(data.values())[0]);
    # Build the beam BREP model
    beam = build_beam(L, A, R, t)
    step_writer = STEPControl_Writer()
    step_writer.Transfer(beam, STEPControl_AsIs)
    status = step_writer.Write("SelectedBeam.step")
    if status != IFSelect_RetDone:
        raise AssertionError("File export failed!")


if __name__ == "__main__":
    app.run(debug=True)
