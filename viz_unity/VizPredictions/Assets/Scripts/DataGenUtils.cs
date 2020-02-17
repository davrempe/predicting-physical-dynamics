using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

public class DataGenUtils
{
	// Converts an OBJMesh (assumed to be a tri-mesh) to a bullet-friendly structure
    public static Mesh ToUnityMesh(OBJLoader.OBJMesh objMesh)
    {
        Mesh mesh = new Mesh();
        UnityEngine.Vector3[] newVertices = new UnityEngine.Vector3[objMesh.faces.Count * 3]; // must make a separate copy of each for normals to be correct
		int[] newTriangles = new int[objMesh.faces.Count * 3];

        for (int i = 0; i < objMesh.vertices.Count; i++) {
            newVertices[i] = objMesh.vertices[i];
        }
        for (int i = 0; i < objMesh.faces.Count; i++) {
			UnityEngine.Vector3 v1 = objMesh.vertices[objMesh.faces[i].indexes[0]];
			UnityEngine.Vector3 v2 = objMesh.vertices[objMesh.faces[i].indexes[1]];
			UnityEngine.Vector3 v3 = objMesh.vertices[objMesh.faces[i].indexes[2]];

			newVertices[i*3 + 0] = v1;
			newVertices[i*3 + 1] = v2;
			newVertices[i*3 + 2] = v3;

            newTriangles[i*3 + 0] = i*3 + 0;
            newTriangles[i*3 + 1] = i*3 + 1;
            newTriangles[i*3 + 2] = i*3 + 2;

        }

        mesh.vertices = newVertices;
        mesh.triangles = newTriangles;
        mesh.RecalculateBounds();
        mesh.RecalculateNormals();

        return mesh;
    }
}
