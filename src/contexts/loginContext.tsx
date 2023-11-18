import { onAuthStateChanged } from "firebase/auth";
import { createSignal, createContext, useContext, ParentComponent, createEffect, Accessor } from "solid-js";
import { auth } from "../firebaseConfig";
import { User } from "firebase/auth";

interface LoginContextData {
  user: User | null,
  loading: boolean
}

const defaultContextData = {
  user: null,
  loading: true,
} as LoginContextData

const LoginContext = createContext<Accessor<LoginContextData>>(() => defaultContextData);

export const LoginProvider: ParentComponent = (props) => {
  const [getContextData, setContextData] = createSignal<LoginContextData>(defaultContextData)
  createEffect(() => {
    onAuthStateChanged(auth, (user) => {
      if (user) {
        setContextData({ user, loading: false } as LoginContextData)
      }
      else {
        setContextData({ user: null, loading: false } as LoginContextData)
      }
    })
  })
  return (
    <LoginContext.Provider value={getContextData}>
      {props.children}
    </LoginContext.Provider>
  );
}

export const useLogin = () => {
  return [useContext(LoginContext)]
}; 
