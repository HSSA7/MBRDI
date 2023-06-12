from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge,
                                     BRepBuilderAPI_MakeFace,
                                     BRepBuilderAPI_MakeWire)
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeSegment
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

from double_hat_solver import get_cross_section_face, get_features_from_json

Interface_Static_SetCVal("write.step.schema", "AP242")

def build_beam(L, A, R, t):
    """
    Given the cross-section parameters, construct the 3D beam.
    """
    # Get the coordinates of the profile points
    face = get_cross_section_face(L, A, R, t, 3, False)

    left_points = []
    for x, y in face[0:26]:
        left_points.append(gp_Pnt(x, y, 0.0))

    right_points = []
    for x, y in face[26:]:
        right_points.append(gp_Pnt(x, y, 0.0))
    
    # Create the left profile
    left_segments = []
    for i in range(0, 25, 3):
        segment = GC_MakeSegment(left_points[i], left_points[i+1])
        left_segments.append(BRepBuilderAPI_MakeEdge(segment.Value()).Edge())

    left_arcs = []
    for i in range(1, 23, 3):
        arc = GC_MakeArcOfCircle(left_points[i], left_points[i+1], left_points[i+2])
        left_arcs.append(BRepBuilderAPI_MakeEdge(arc.Value()).Edge())

    profile_edges = []
    for line, arc in zip(left_segments, left_arcs):
        profile_edges.append(line)
        profile_edges.append(arc)
    
    profile_edges.append(left_segments[-1])

    # Create the bottom connector
    bottom_segment = GC_MakeSegment(left_points[-1], right_points[0])
    profile_edges.append(BRepBuilderAPI_MakeEdge(bottom_segment.Value()).Edge())

    # Create the right profile
    right_segments = []
    for i in range(0, 25, 3):
        segment = GC_MakeSegment(right_points[i], right_points[i+1])
        right_segments.append(BRepBuilderAPI_MakeEdge(segment.Value()).Edge())

    right_arcs = []
    for i in range(1, 23, 3):
        arc = GC_MakeArcOfCircle(right_points[i], right_points[i+1], right_points[i+2])
        right_arcs.append(BRepBuilderAPI_MakeEdge(arc.Value()).Edge())

    for line, arc in zip(right_segments, right_arcs):
        profile_edges.append(line)
        profile_edges.append(arc)

    profile_edges.append(right_segments[-1])

    # Create the top connector
    top_segment = GC_MakeSegment(right_points[-2], right_points[-1])
    profile_edges.append(BRepBuilderAPI_MakeEdge(top_segment.Value()).Edge())

    # Create the profile wire
    profile_wire = BRepBuilderAPI_MakeWire()
    for edge in profile_edges:
        profile_wire.Add(BRepBuilderAPI_MakeWire(edge).Wire())

    # Create the face
    profile_face = BRepBuilderAPI_MakeFace(profile_wire.Wire())

    # Extrude length
    extrude_vec_1 = gp_Vec(0.0, 0.0, -0.5*L[-1])
    extrude_vec_2 = gp_Vec(0.0, 0.0, 0.5*L[-1])

    # Make the beam
    half_beam_1 = BRepPrimAPI_MakePrism(profile_face.Face(), extrude_vec_1).Shape()
    half_beam_2 = BRepPrimAPI_MakePrism(profile_face.Face(), extrude_vec_2).Shape()
    beam = BRepAlgoAPI_Fuse(half_beam_1, half_beam_2).Shape()

    return beam
    

def get_volume(beam):
    """
    Given a TopoDS_Shape object of a beam, return the volume.
    """
    system = GProp_GProps()
    brepgprop_VolumeProperties(beam, system)
    volume = system.Mass()
    return volume
    

def export_to_step(beam, filename='SelectedBeam.step'):
    """
    Export STEP file
    """
    step_writer = STEPControl_Writer()
    step_writer.Transfer(beam, STEPControl_AsIs)
    status = step_writer.Write(filename)

    if status != IFSelect_RetDone:
        raise AssertionError("File export failed!")


if __name__ == "__main__":
    from OCC.Display.SimpleGui import init_display

    # Build beam
    L, A, R, t = get_features_from_json('biw_single_part/sample_data/input_dh_73.json')
    beam = build_beam(L, A, R, t)
    print('The volume of beam calculated using Opencascade is', get_volume(beam))


    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(beam, update=True)
    start_display()
