import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';

@Component({
  selector: 'app-tag-generator',
  templateUrl: './tag-generator.component.html',
  styleUrls: ['./tag-generator.component.css']
})
export class TagGeneratorComponent implements OnInit {

  file!: File;

  constructor(private http: HttpClient, private router:Router) { }

  ngOnInit(): void {
  }

  OnTagChange(event: any){
    this.file= event.target.files[0];
    console.log(this.file)
  }
  OnTagSubmit(){
    const uploadData =new FormData();
    uploadData.append('file',this.file);

    this.http.post('http://127.0.0.1:5000/api/upload/tagimage',uploadData).subscribe(
      data => console.log(data),
      error => console.log(error)
    );
   
  }

}
