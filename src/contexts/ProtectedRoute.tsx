import { ParentComponent, createEffect } from "solid-js";
import { useLogin } from "./loginContext";
import Spinner from "../components/Spinner";
import { Navigate } from "@solidjs/router";

const ProtectedRoute: ParentComponent = (props) => {
  const [getLoginContext] = useLogin()
  if(getLoginContext().loading)
    return <Spinner/>
  return getLoginContext().user ? props.children : <Navigate href="./"/>
} 

export default ProtectedRoute;