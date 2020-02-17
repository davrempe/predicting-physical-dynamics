using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class TumbleVizPred : MonoBehaviour {


    //
    // Publicly exposed members (in the editor).
    //

    [Tooltip("Ensures paths are set w.r.t. Builds directory rather than Assets/Scripts")]
    public bool BUILD_MODE = false;

    [Tooltip("Directory of .json prediction files. Should be relative path from Assets directory of project.")]
    public string [] predDataDirs = { "../../DataIn/Predictions/CubeExample/pred_out" };
    [Tooltip("Directories to find object meshes for predictions. Should be relative path from Assets directory of project.")]
    public string[] meshSources = { "../../DataIn/Cubes/CubeAll" };
    [Tooltip("Will write out images of prediction sequences here. Should be relative path from Assets directory of project.")]
    public string dataOutDir = "../../DataOut/cube_ex";
    [Tooltip("Randomly order shown predictions.")]
    public bool shufflePredictions = true;
    [Tooltip("Only show predictions sequences with no toppling in original simulation.")]
    public bool showNonTopplingOnly = false;
    [Tooltip("Only show predictions sequences with toppling in original simulation.")]
    public bool showTopplingOnly = false;
    [Tooltip("In not -1, only the specific sim index will be shown.")]
    public int onlyShowSimIdx = -1;
    [Tooltip("Only show predicted sequences.")]
    public bool hideGroundTruth = false;
    [Tooltip("Shows rotation axes in the scene view (not the rendered view).")]
    public bool drawAxis = false;

    // object appearance
    public Material groundMat;
    public Material gtMat;
    public Material[] predMats;

    public Vector3 perspCameraLoc = new Vector3(1, 1, -4);
    public Vector3 perspCameraTarget = new Vector3(0, 0, 0);
    public Vector3 placeOffset = new Vector3(0, 0, 0);

    // directories of prediction files
    protected List<string> m_predDataDirs;
    // directories for all possible meshes we could need
    protected List<string> m_baseModelDirs;
    // save videos base directories
    protected string m_imgOutDir;
    // current simulation video dir
    protected string m_curSimOutDir;
    // list of sim indices to visualize
    List<int> m_simIdxList;
    // the sim we're on
    protected int m_simNum;
    protected int m_frameNum;
    protected int m_totalFrames;

    protected Mesh m_mesh;
    protected GameObject m_gtObj;
    protected GameObject groundObj;
    protected List<GameObject> m_predObjs;

    protected Color ogColor;

    private List<PredInfo> m_curPreds;

    protected bool firstFrame;

    // Use this for initialization
    void Start() {

        string baseDir = Application.dataPath;
        if (BUILD_MODE) {
            baseDir = Path.Combine(Application.dataPath, "../../");
        }

        m_predDataDirs = new List<string>(predDataDirs.Length);
        // directory to find prediction .json files
        for (int i = 0; i < predDataDirs.Length; i++) {
            string predDataDir = Path.Combine(baseDir, predDataDirs[i]);
            m_predDataDirs.Add(predDataDir);
        }
        // assuming want to viz from same data
        DirectoryInfo sourceDirInfo = new DirectoryInfo(m_predDataDirs[0]);
        int numPreds = sourceDirInfo.GetFiles().Length;
        IEnumerable<int> idxList = Enumerable.Range(0, numPreds);
        if (shufflePredictions) {
            // randomize their order
            m_simIdxList = idxList.OrderBy(x => Random.value).ToList();
        } else {
            m_simIdxList = idxList.ToList();
        }

        if (onlyShowSimIdx != -1) {
            m_simIdxList = new List<int>();
            m_simIdxList.Add(onlyShowSimIdx);
        }

        // directories to look for meshes
        m_baseModelDirs = new List<string>(meshSources.Length);
        for (int i = 0; i < meshSources.Length; i++) {
            m_baseModelDirs.Add(Path.Combine(baseDir, meshSources[i]));
        }
        // base directory to save videos
        m_imgOutDir = Path.Combine(baseDir, dataOutDir);
        Directory.CreateDirectory(m_imgOutDir);

        SetEyeTarget(perspCameraLoc, perspCameraTarget);

        m_curPreds = new List<PredInfo>(predDataDirs.Length);
        m_predObjs = new List<GameObject>(predDataDirs.Length);

        // create ground
        groundObj = GameObject.CreatePrimitive(PrimitiveType.Plane);
        groundObj.transform.localScale = new Vector3(20.0f, 1.0f, 20.0f);
        groundObj.transform.Rotate(new Vector3(0.0f, 1.0f, 0.0f), 0.0f);
        groundObj.GetComponent<MeshRenderer>().sharedMaterial = groundMat;
        ogColor = groundMat.color;

        firstFrame = true;

        // set up first simulation
        m_simNum = 0;
        bool useNext = false;
        while (!useNext) {
            ResetSim(m_simIdxList[m_simNum]);
            useNext = LoadSim(0, m_simIdxList[m_simNum], true);
            if (useNext) {
                for (int i = 1; i < predDataDirs.Length; i++) {
                    LoadSim(i, m_simIdxList[m_simNum], false);
                }
            } else {
                m_predObjs.Clear();
                m_curPreds.Clear();
            }
            if (!useNext) m_simNum++;
        }
        PlaceCamera();
    }

    void ResetSim(int nextSimNum) {
        m_frameNum = 1;
        m_curSimOutDir = Path.Combine(m_imgOutDir, "sim_" + nextSimNum.ToString());
        Directory.CreateDirectory(m_curSimOutDir);
    }

    // returns false if we don't want to use this sim
    bool LoadSim(int predIdx, int idx, bool createGt) {
        // read in results from json
        string predFile = Path.Combine(m_predDataDirs[predIdx], "eval_sim_" + idx.ToString() + ".json");
        Debug.Log("Visualizing " + predFile);
        string jsonTextFile = ReadFile(predFile);
        m_curPreds.Add(PredInfo.CreateFromJSON(jsonTextFile));

        if (showTopplingOnly && !m_curPreds[predIdx].toppled) {
            // don't want to use this
            return false;
        }

        if (showNonTopplingOnly && m_curPreds[predIdx].toppled) {
            return false;
        }

        // find mesh to use
        string meshFile = "";
        for (int i = 0; i < m_baseModelDirs.Count; i++) {
            meshFile = Path.Combine(m_baseModelDirs[i], m_curPreds[predIdx].shape);
            meshFile += ".obj";
            if (File.Exists(meshFile)) {
                break;
            }
        }
        Debug.Log(meshFile);
        if (meshFile == "") {
            Debug.Log("Couldn't find mesh " + m_curPreds[predIdx].shape + "!");
            Application.Quit();
        }

        OBJLoader.OBJMesh objLoaderMesh = OBJLoader.LoadOBJMesh(meshFile);
        Debug.Assert(objLoaderMesh.vertices.Count > 0);
        Debug.Assert(objLoaderMesh.faces.Count > 0);
        m_mesh = DataGenUtils.ToUnityMesh(objLoaderMesh);

        // create GT object
        if (createGt && !hideGroundTruth) {
            m_gtObj = new GameObject("Sim" + idx.ToString() + "_GT");
            MeshFilter mf = m_gtObj.AddComponent<MeshFilter>();
            mf.mesh = m_mesh;
            MeshRenderer mr = m_gtObj.AddComponent<MeshRenderer>();
            mr.sharedMaterial = gtMat;
            m_gtObj.transform.localScale = m_curPreds[predIdx].scale;
            m_gtObj.transform.position = m_curPreds[predIdx].gt_pos[0];
            m_gtObj.transform.eulerAngles = m_curPreds[predIdx].gt_euler_rot[0];
        }

        // create sampled object
        m_predObjs.Add(new GameObject("Sim" + idx.ToString() + "_Pred"));
        MeshFilter mf2 = m_predObjs[predIdx].AddComponent<MeshFilter>();
        mf2.mesh = m_mesh;
        MeshRenderer mr2 = m_predObjs[predIdx].AddComponent<MeshRenderer>();
        mr2.sharedMaterial = predMats[predIdx];
        m_predObjs[predIdx].transform.localScale = m_curPreds[predIdx].scale;
        m_predObjs[predIdx].transform.position = m_curPreds[predIdx].pred_pos[0];
        m_predObjs[predIdx].transform.eulerAngles = m_curPreds[predIdx].pred_rot[0];

        return true;
    }

    // Update is called once per frame
    void Update() {
        if (firstFrame) {
            // can't saved screenshots on first frame
            firstFrame = false;
            return;
        }

        // check if we're done with the current simulation
        int stepNum = m_frameNum - 1;
        if (stepNum >= m_curPreds[0].pred_pos.Count) {
            if (!hideGroundTruth) Destroy(m_gtObj);
            for (int i = 0; i < m_predObjs.Count(); i++) {
                Destroy(m_predObjs[i]);
            }
            Destroy(m_mesh);

            m_predObjs.Clear();
            m_curPreds.Clear();

            // set up next sim
            bool useNextSim = false;
            while (!useNextSim) {
                m_simNum++;
                if (m_simNum >= m_simIdxList.Count) {
# if UNITY_EDITOR
                    UnityEditor.EditorApplication.isPlaying = false;
#else
                    Application.Quit();
#endif
                }

                ResetSim(m_simIdxList[m_simNum]);
                // load the first one to see if we want to use it
                useNextSim = LoadSim(0, m_simIdxList[m_simNum], true);
                if (useNextSim) {
                    for (int i = 1; i < predDataDirs.Length; i++) {
                        LoadSim(i, m_simIdxList[m_simNum], false);
                    }
                } else {
                    m_predObjs.Clear();
                    m_curPreds.Clear();
                }
            }

            // now set up the camera to be a good view of this sim base on final GT state
            PlaceCamera();
        } else {
            // set up next frame
            Vector3 gt_rot_axis = new Vector3();
            float gt_rot_angle = 0.0f;
            if (!hideGroundTruth) {
                m_gtObj.transform.position = m_curPreds[0].gt_pos[stepNum];
                m_gtObj.transform.eulerAngles = m_curPreds[0].gt_euler_rot[stepNum];

                // load rotation axes so they can be visualized if desired 
                gt_rot_axis = (m_curPreds[0].gt_delta_rot[stepNum] / m_curPreds[0].gt_delta_rot[stepNum].magnitude);
                gt_rot_angle = m_curPreds[0].gt_delta_rot[stepNum].magnitude;
                if (gt_rot_axis.Equals(Vector3.zero)) {
                    gt_rot_axis = new Vector3(1.0f, 0.0f, 0.0f);
                }
            }

            if (drawAxis) {
                if (!hideGroundTruth) {
                    Debug.DrawRay(m_curPreds[0].gt_pos[stepNum], gt_rot_axis * (gt_rot_angle + 1.0f) * 0.05f,m_gtObj.GetComponent<MeshRenderer>().sharedMaterial.color, 1.0f);
                }
            }

            for (int i = 0; i < m_curPreds.Count(); i++) {
                m_predObjs[i].transform.position = m_curPreds[i].pred_pos[stepNum];

                Vector3 curRot = m_curPreds[i].pred_rot[stepNum];
                m_predObjs[i].transform.eulerAngles = curRot;

                // load rotation axis and angle for visualization
                Vector3 rot_axis = new Vector3();
                float rot_angle = -1;

                rot_axis = m_curPreds[i].pred_delta_rot[stepNum];
                rot_angle = rot_axis.magnitude;
                rot_axis = m_curPreds[i].pred_delta_rot[stepNum];
                if (rot_axis.Equals(Vector3.zero)) {
                    rot_axis = new Vector3(1.0f, 0.0f, 0.0f);
                }
                rot_axis.Normalize();

                if (drawAxis) {
                    Debug.DrawRay(m_curPreds[i].pred_pos[stepNum], rot_axis * (rot_angle + 1.0f) * 0.05f, m_predObjs[i].GetComponent<MeshRenderer>().sharedMaterial.color, 1.0f);
                }
            }
        }

        SaveFrame(m_frameNum);
        m_frameNum++;
    }

    protected void SaveFrame(int frameIdx) {
        string outFile = Path.Combine(m_curSimOutDir, "frame_" + frameIdx.ToString().PadLeft(6, '0') + ".png");
        ScreenCapture.CaptureScreenshot(outFile, 1);
    }

    protected void PlaceCamera() {
        int finalFrame = m_curPreds[0].gt_pos.Count - 1;
        Vector3 finalPos = m_curPreds[0].gt_pos[finalFrame];
        Vector3 zDir = finalPos.normalized;
        zDir.y = 0.0f;
        Vector3 xDir = new Vector3(zDir.z, 0.0f, -zDir.x);
        Vector3 yDir = new Vector3(0.0f, 1.0f, 0.0f);
        Vector3 offset = xDir * placeOffset.x + yDir * placeOffset.y + zDir * placeOffset.z;

        finalPos.y = 0.0f;
        Vector3 targ = Vector3.zero;
        targ.y = 0.0f;
        SetEyeTarget(finalPos + offset, targ);
    }

    protected void SetEyeTarget(Vector3 eye, Vector3 targ) {
        Transform t = Camera.main.transform;
        t.position = eye;
        t.rotation = Quaternion.LookRotation((targ - eye).normalized, Vector3.up);
    }

    // Reads an entire file at the given path to a string.
    protected string ReadFile(string filePath) {
        string readContents;
        using (StreamReader streamReader = new StreamReader(filePath)) {
            readContents = streamReader.ReadToEnd();
        }

        return readContents;
    }

    // Object for reading from prediction info json
    private class PredInfo {
        public string shape;
        public Vector3 scale;
        public List<Vector3> gt_pos;
        public List<Vector3> gt_total_rot;
        public List<Vector3> gt_euler_rot;
        public List<Vector3> gt_delta_rot;
        public List<Quaternion> gt_delta_quat;
        public List<Vector3> pred_pos;
        public List<Vector3> pred_rot;
        public List<Vector3> pred_delta_rot;
        public List<Quaternion> pred_delta_quat;
        public float pos_err;
        public float rot_err;
        public bool toppled;
        public float body_friction;

        public static PredInfo CreateFromJSON(string jsonString) {
            return JsonUtility.FromJson<PredInfo>(jsonString);
        }
    }
}
