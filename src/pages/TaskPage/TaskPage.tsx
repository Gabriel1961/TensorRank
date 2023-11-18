import { A, useParams } from "@solidjs/router";
import { Component, Show, createEffect, createSignal, onCleanup, onMount } from "solid-js";
import styles from "./Task.module.scss"
import * as monaco from 'monaco-editor';
import { DatasetData, Task, decodeTextTitleToUrl } from "../../datamodel";
import * as tf from "@tensorflow/tfjs"
import { VsBracketError } from 'solid-icons/vs'
import { fetchTaskAfterTitle, fetchTaskData, uploadSolution } from "./TaskPageFetcher";
import { TaskTable } from "./components/TaskTable";
import TaskTest, { TestEval } from "./components/TaskTest";
import { evalModel } from "./TaskEval";
import { createStore, produce } from "solid-js/store";
import { useLogin } from "../../contexts/loginContext";

interface TaskPageParams {
  taskName: string
}

const TaskPage: Component = () => {
  const params = useParams() as any as TaskPageParams
  const [getLoginContext] = useLogin()
  const [getEditor, setEditor] = createSignal<monaco.editor.IStandaloneCodeEditor | null>(null)
  const [getTextModel, setTextModel] = createSignal<monaco.editor.ITextModel | null>(null)
  const [getTask, setTask] = createSignal<Task | null>(null)
  const [getDatasetData, setDatasetData] = createSignal<DatasetData | null>(null)
  const [getErrorMessage, setErrorMessage] = createSignal<string>("")
  const [getTaskEvals, setTaskEvals] = createStore<TestEval[]>([])

  let editor: HTMLIFrameElement

  onMount(async () => {
    // load editor 
    const libUri = "ts:filename/tfjs.d.ts";
    const tfjsSource = await (await fetch("/src/assets/tfjs.txt")).text()
    const defaultCode = await (await fetch("/src/assets/defaultCode.txt")).text()
    monaco.languages.typescript.javascriptDefaults.addExtraLib(tfjsSource, libUri);
    const textModel = monaco.editor.createModel(tfjsSource, "typescript", monaco.Uri.parse(libUri));

    setTextModel(textModel)

    const editorInstance = monaco.editor.create(editor, {
      value: defaultCode,
      language: "javascript",
      theme: "vs-dark"
    });

    setEditor(editorInstance)
  })

  onCleanup(() => {
    const textModel = getTextModel()
    const editor = getEditor()
    textModel?.dispose()
    editor?.dispose()
  })

  createEffect(() => {
    fetchTaskAfterTitle(decodeTextTitleToUrl(params.taskName)).then(task => {
      if (task) {
        setTask(task)
        fetchTaskData(task)
          .then(data => {
            setDatasetData(data)
          })
      }
    })
  })

  const onRunClick = async () => {
    if (!getDatasetData() || !getTask())
      return
    if (!getLoginContext().user) {
      alert("Please login :)")
      return 
    }

    // required 
    const wat = tf.Abs
    console.log(wat + "Started tf")

    const editor = getEditor()

    var model = null
    if (editor) {
      try {
        const sourceCode = editor.getValue()
        model = eval(editor.getValue())

        // eval the model 
        const newEvals: TestEval[] = []
        const numberOfTests = 6;

        for (let i = 0; i < numberOfTests; i++)
          newEvals.push({ accuracy: 0, done: false } as TestEval)
        setTaskEvals(newEvals)

        let accMed = 0
        for (let i = 0; i < numberOfTests; i++) {
          const accuracy = await evalModel(getDatasetData()!, model)
          setTaskEvals(produce(evals => {
            evals[i] = { accuracy, done: true } as TestEval
          }))
          accMed += accuracy
        }

        accMed /= numberOfTests

        uploadSolution(accMed, getTask()!.id,getLoginContext()!.user!,sourceCode)
      }
      catch (ex) {
        const error = ex as Error
        setErrorMessage(error.message)
      }
    }
  }

  return <div class={styles.container}>
    <div class={styles.header}>
      <h2 class={styles.h2}>{decodeTextTitleToUrl(params.taskName)}</h2>
      <A href="/scoreboard">
        <h2 class={styles.h2}>Scoreboard</h2>
      </A>
    </div>
    <TaskTable dataset={getDatasetData()} />
    <div ref={editor} class={styles.codeEditor} />
    <button onClick={onRunClick}>Run</button>
    <Show when={getErrorMessage()}>
      <div class={styles.errorContainer}>
        <VsBracketError color={"#e74c3c"} style={{ "margin-right": "10px" }} size={30} />
        <p>
          {getErrorMessage()}
        </p>
      </div>
    </Show>
    <TaskTest tests={getTaskEvals} />
  </div>
}

export default TaskPage;